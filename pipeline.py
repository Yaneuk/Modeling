import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch
import torch.nn.functional as F


# =============================================
# 경로 설정
# =============================================
CLS_MODEL_PATH = "/workspace/results/classification/cls_v1/weights/best.pt"
DET_MODEL_PATH = "/workspace/results/detection/det_v1/weights/best.pt"

CLASS_NAMES = {0: "reusable", 1: "recycle", 2: "dispose"}

# 2단계로 넘어갈 클래스
DEFECT_CLASSES = {"recycle", "dispose"}

# 신뢰도 임계값
CLS_CONF_THRESHOLD = 0.6
DET_CONF_THRESHOLD = 0.4


# =============================================
# 모델 로드
# =============================================
print("[로드] 분류 모델...")
cls_model = YOLO(CLS_MODEL_PATH)

print("[로드] 탐지 모델...")
det_model = YOLO(DET_MODEL_PATH)

print("[준비 완료]\n")


# =============================================
# GradCAM 생성 함수
# =============================================
def generate_gradcam(image_path: str, class_idx: int) -> np.ndarray:
    """
    분류 모델의 판단 근거를 히트맵으로 생성

    Args:
        image_path: 이미지 경로
        class_idx:  분류된 클래스 번호

    Returns:
        히트맵이 원본 이미지에 합성된 BGR 이미지 (numpy array)
    """
    # 내부 PyTorch 모델 추출
    model = cls_model.model
    model.eval()

    # 이미지 전처리
    img_bgr  = cv2.imread(image_path)
    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_tensor  = torch.tensor(
        img_resized.transpose(2, 0, 1), dtype=torch.float32
    ).unsqueeze(0) / 255.0

    # 마지막 Conv 레이어에 훅 등록
    gradients  = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # 마지막 Conv 레이어 찾기
    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module

    if target_layer is None:
        return img_bgr

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    # Forward + Backward
    output = model(img_tensor)
    model.zero_grad()

    if hasattr(output, 'logits'):
        score = output.logits[0, class_idx]
    else:
        score = output[0, class_idx]

    score.backward()

    fh.remove()
    bh.remove()

    if not gradients or not activations:
        return img_bgr

    # GradCAM 계산
    grad = gradients[0].detach()
    act  = activations[0].detach()

    weights  = grad.mean(dim=(2, 3), keepdim=True)
    cam      = (weights * act).sum(dim=1).squeeze()
    cam      = F.relu(cam)

    # 정규화
    cam = cam.numpy()
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    # 원본 이미지 크기로 리사이즈
    h, w = img_bgr.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # 히트맵 컬러 적용
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )

    # 원본 이미지와 합성
    overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

    return overlay


# =============================================
# 파이프라인 메인 함수
# =============================================
def run_pipeline(image_path: str, save_dir: str = None) -> dict:
    """
    이미지 1장에 대해 2단계 파이프라인 + GradCAM 실행

    Returns:
        {
            "file": 파일명,
            "stage1": {
                "class": 분류 결과,
                "confidence": 신뢰도
            },
            "stage2": {
                "defects": [
                    {
                        "class": 클래스명,
                        "confidence": 신뢰도,
                        "bbox": [x1, y1, x2, y2]
                    }
                ]
            },
            "gradcam_path": GradCAM 이미지 저장 경로
        }
    """
    image_path = str(image_path)
    filename   = os.path.basename(image_path)
    result     = {"file": filename, "stage1": {}, "stage2": None, "gradcam_path": None}

    # ── 1단계: 분류 ───────────────────────────
    cls_result = cls_model.predict(image_path, verbose=False)[0]
    top1_idx   = cls_result.probs.top1
    top1_conf  = float(cls_result.probs.top1conf)
    top1_name  = CLASS_NAMES.get(top1_idx, str(top1_idx))

    result["stage1"] = {"class": top1_name, "confidence": round(top1_conf, 4)}
    print(f"[1단계] {filename} → {top1_name} (신뢰도: {top1_conf:.2%})")

    # ── GradCAM 생성 (항상 생성) ──────────────
    if save_dir:
        try:
            gradcam_img  = generate_gradcam(image_path, top1_idx)
            gradcam_path = os.path.join(save_dir, f"gradcam_{filename}")
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(gradcam_path, gradcam_img)
            result["gradcam_path"] = gradcam_path
            print(f"  → GradCAM 저장: {gradcam_path}")
        except Exception as e:
            print(f"  → GradCAM 생성 실패: {e}")

    # reusable 또는 신뢰도 낮으면 탐지 생략
    if top1_name not in DEFECT_CLASSES or top1_conf < CLS_CONF_THRESHOLD:
        print(f"  → 정상 판정, 탐지 생략\n")
        return result

    # ── 2단계: 탐지 ───────────────────────────
    print(f"  → 하자 의심, 탐지 시작...")
    det_results = det_model.predict(
        image_path,
        conf=DET_CONF_THRESHOLD,
        verbose=False
    )[0]

    defects = []
    if det_results.boxes is not None:
        for box in det_results.boxes:
            cls_id        = int(box.cls)
            conf          = float(box.conf)
            x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0]]
            defects.append({
                "class":      CLASS_NAMES.get(cls_id, str(cls_id)),
                "confidence": round(conf, 4),
                "bbox":       [x1, y1, x2, y2]
            })

    result["stage2"] = {"defects": defects}
    print(f"  → 탐지된 하자: {len(defects)}개\n")

    # ── 결과 이미지 저장 ──────────────────────
    if save_dir:
        _save_annotated(image_path, det_results, save_dir, filename)

    return result


def _save_annotated(image_path, det_results, save_dir, filename):
    """바운딩박스가 그려진 결과 이미지 저장"""
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(image_path)

    color_map = {
        "reusable": (0, 200, 0),
        "recycle":  (0, 165, 255),
        "dispose":  (0, 0, 220),
    }

    if det_results.boxes is not None:
        for box in det_results.boxes:
            cls_id        = int(box.cls)
            conf          = float(box.conf)
            cls_name      = CLASS_NAMES.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            color         = color_map.get(cls_name, (200, 200, 200))
            label         = f"{cls_name} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_path = os.path.join(save_dir, f"result_{filename}")
    cv2.imwrite(out_path, img)
    print(f"  → 결과 이미지 저장: {out_path}")


# =============================================
# 폴더 일괄 처리
# =============================================
def run_pipeline_batch(input_dir: str, save_dir: str = None) -> list:
    """폴더 안의 모든 이미지에 파이프라인 실행"""
    image_exts  = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [
        p for p in Path(input_dir).iterdir()
        if p.suffix.lower() in image_exts
    ]

    print(f"[배치] 총 {len(image_paths)}장 처리 시작\n")

    all_results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}]")
        res = run_pipeline(str(img_path), save_dir=save_dir)
        all_results.append(res)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_json = os.path.join(save_dir, "pipeline_results.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[완료] 결과 저장: {out_json}")

    total     = len(all_results)
    reusable  = sum(1 for r in all_results if r["stage1"]["class"] == "reusable")
    defective = total - reusable
    print(f"\n[요약]")
    print(f"  전체:   {total}장")
    print(f"  정상:   {reusable}장 ({reusable/total:.1%})")
    print(f"  하자:   {defective}장 ({defective/total:.1%})")

    return all_results


# =============================================
# 실행 진입점
# =============================================
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 2:
        image_path = sys.argv[1]
        result = run_pipeline(image_path, save_dir="/workspace/results/pipeline")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    elif len(sys.argv) == 3 and sys.argv[1] == "--batch":
        input_dir = sys.argv[2]
        run_pipeline_batch(input_dir, save_dir="/workspace/results/pipeline")

    else:
        print("사용법:")
        print("  단일 이미지: python pipeline.py 이미지경로.jpg")
        print("  폴더 일괄:   python pipeline.py --batch 폴더경로")
