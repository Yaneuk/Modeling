import json
import os
from pathlib import Path


# =============================================
# 클래스 매핑
# big_category 값 → YOLO 클래스 번호
# =============================================
CLASS_MAP = {
    "reusable": 0,
    "recycle":  1,
    "dispose":  2,
}


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    커스텀 JSON bbox [x_min, y_min, x_max, y_max]
    → YOLO 형식 [cx, cy, w, h] (0~1 정규화)
    """
    x_min, y_min, x_max, y_max = bbox

    cx = (x_min + x_max) / 2 / img_width
    cy = (y_min + y_max) / 2 / img_height
    w  = (x_max - x_min) / img_width
    h  = (y_max - y_min) / img_height

    # 0~1 범위 클램핑 (이상값 방지)
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    return cx, cy, w, h


def convert_single_json(json_path, output_dir):
    """
    JSON 파일 1개를 YOLO txt 파일로 변환

    변환 규칙:
    - big_category → 클래스 번호
    - annotations.pollution[].bbox → 하자 위치
    - annotations.clothes[].bbox  → 옷 전체 영역 (pollution 없을 때 폴백)
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    meta        = data["meta_information"]
    img_width   = meta["width"]
    img_height  = meta["height"]
    big_cat     = meta["big_category"].lower().strip()
    file_stem   = Path(meta["file_name"]).stem  # 확장자 제거

    cls_id = CLASS_MAP.get(big_cat)
    if cls_id is None:
        print(f"[SKIP] 알 수 없는 big_category '{big_cat}' → {json_path.name}")
        return False

    annotations = data.get("annotations", {})
    pollution   = annotations.get("pollution", [])
    clothes     = annotations.get("clothes", [])

    lines = []

    # ── 하자 영역 변환 ──────────────────────────
    if pollution:
        for item in pollution:
            for bbox in item["bbox"]:
                cx, cy, w, h = convert_bbox_to_yolo(bbox, img_width, img_height)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # ── pollution 없으면 옷 전체 영역으로 폴백 ──
    else:
        for item in clothes:
            for bbox in item["bbox"]:
                cx, cy, w, h = convert_bbox_to_yolo(bbox, img_width, img_height)
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # ── txt 저장 ────────────────────────────────
    out_path = Path(output_dir) / f"{file_stem}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return True


def convert_all(json_dir, output_dir):
    """
    json_dir 안의 모든 JSON 파일을 일괄 변환

    Args:
        json_dir   : JSON 파일들이 있는 폴더 경로
        output_dir : 변환된 txt 파일을 저장할 폴더 경로
    """
    json_dir   = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        print(f"[ERROR] JSON 파일을 찾을 수 없습니다: {json_dir}")
        return

    success = 0
    skip    = 0

    for json_path in json_files:
        ok = convert_single_json(json_path, output_dir)
        if ok:
            success += 1
        else:
            skip += 1

    print(f"\n변환 완료")
    print(f"  성공: {success}개")
    print(f"  스킵: {skip}개")
    print(f"  저장 위치: {output_dir}")


# =============================================
# 변환 결과 검증 (샘플 출력)
# =============================================
def verify(output_dir, n=5):
    """변환된 txt 파일 샘플 n개를 출력해서 확인"""
    txt_files = list(Path(output_dir).glob("*.txt"))
    print(f"\n── 검증 샘플 (최대 {n}개) ──────────────────")
    for txt in txt_files[:n]:
        print(f"\n{txt.name}:")
        print(txt.read_text(encoding="utf-8") or "  (빈 파일)")
    print("────────────────────────────────────────")


# =============================================
# 실행 진입점
# =============================================
if __name__ == "__main__":
    # ↓↓ 경로만 본인 환경에 맞게 수정하세요 ↓↓
    JSON_DIR   = "/workspace/dataset/02.라벨링데이터"   # JSON 파일 폴더
    OUTPUT_DIR = "/workspace/dataset/labels_yolo"   # 변환된 txt 저장 폴더

    convert_all(JSON_DIR, OUTPUT_DIR)
    verify(OUTPUT_DIR)
