import os
import cv2
import json
from pathlib import Path
from ultralytics import YOLO


# =============================================
# 경로 설정
# =============================================
CLS_MODEL_PATH = "/workspace/results/classification/cls_v1/weights/best.pt"
DET_MODEL_PATH = "/workspace/results/detection/det_v1/weights/best.pt"

CLASS_NAMES = {0: "reusable", 1: "recycle", 2: "dispose"}

# 2단계로 넘어갈 클래스 (reusable은 정상이므로 탐지 생략)
DEFECT_CLASSES = {"recycle", "dispose"}

# 분류 신뢰도 임계값
CLS_CONF_THRESHOLD = 0.6
# 탐지 신뢰도 임계값
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
# 파이프라인 메인 함수
# =============================================
def run_pipeline(image_path: str, save_dir: str = None) -> dict:
    """
    이미지 1장에 대해 2단계 파이프라인 실행

    Returns:
        {
            "file": 파일명,
            "stage1": {
                "class": 분류 결과,
                "confidence": 신뢰도
            },
            "stage2": {                        # reusable이면 None
                "defects": [
                    {
                        "class": 클래스명,
                        "confidence": 신뢰도,
                        "bbox": [x1, y1, x2, y2]  # 픽셀 좌표
                    },
                    ...
                ]
            }
        }
    """
    image_path = str(image_path)
    filename   = os.path.basename(image_path)
    result     = {"file": filename, "stage1": {}, "stage2": None}

    # ── 1단계: 분류 ───────────────────────────
    cls_result  = cls_model.predict(image_path, verbose=False)[0]
    top1_idx    = cls_result.probs.top1
    top1_conf   = float(cls_result.probs.top1conf)
    top1_name   = CLASS_NAMES.get(top1_idx, str(top1_idx))

    result["stage1"] = {"class": top1_name, "confidence": round(top1_conf, 4)}
    print(f"[1단계] {filename} → {top1_name} (신뢰도: {top1_conf:.2%})")

    # reusable 또는 신뢰도 낮으면 종료
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
    for box in det_results.boxes:
        cls_id   = int(box.cls)
        conf     = float(box.conf)
        x1, y1, x2, y2 = [round(float(v), 2) for v in box.xyxy[0]]
        defects.append({
            "class":      CLASS_NAMES.get(cls_id, str(cls_id)),
            "confidence": round(conf, 4),
            "bbox":       [x1, y1, x2, y2]
        })

    result["stage2"] = {"defects": defects}
    print(f"  → 탐지된 하자: {len(defects)}개\n")

    # ── 결과 이미지 저장 (선택) ────────────────
    if save_dir:
        _save_annotated(image_path, det_results, save_dir, filename)

    return result


def _save_annotated(image_path, det_results, save_dir, filename):
    """바운딩박스가 그려진 결과 이미지 저장"""
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(image_path)

    color_map = {
        "reusable": (0, 200, 0),    # 초록
        "recycle":  (0, 165, 255),  # 주황
        "dispose":  (0, 0, 220),    # 빨강
    }

    for box in det_results.boxes:
        cls_id       = int(box.cls)
        conf         = float(box.conf)
        cls_name     = CLASS_NAMES.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        color        = color_map.get(cls_name, (200, 200, 200))
        label        = f"{cls_name} {conf:.2f}"

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
    """
    폴더 안의 모든 이미지에 파이프라인 실행
    결과를 JSON 파일로 저장
    """
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
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

    # JSON으로 결과 저장
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_json = os.path.join(save_dir, "pipeline_results.json")
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[완료] 결과 저장: {out_json}")

    # 간단 요약
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

    # 단일 이미지 테스트
    # python pipeline.py 이미지경로.jpg
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
        result = run_pipeline(image_path, save_dir="/workspace/results/pipeline")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    # 폴더 일괄 처리
    # python pipeline.py --batch 폴더경로
    elif len(sys.argv) == 3 and sys.argv[1] == "--batch":
        input_dir = sys.argv[2]
        run_pipeline_batch(input_dir, save_dir="/workspace/results/pipeline")

    else:
        print("사용법:")
        print("  단일 이미지: python pipeline.py 이미지경로.jpg")
        print("  폴더 일괄:   python pipeline.py --batch 폴더경로")
