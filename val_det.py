import os
from datetime import datetime
from ultralytics import YOLO


# =============================================
# 경로 설정
# =============================================
MODEL_PATH = "/workspace/results/detection/det_v1/weights/best.pt"
YAML_PATH  = "/workspace/Modeling/data_det.yaml"
RESULT_DIR = "/workspace/results/detection/det_v1/val"

os.makedirs(RESULT_DIR, exist_ok=True)

# =============================================
# 검증 실행
# =============================================
start_time = datetime.now()
print(f"[시작] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

model   = YOLO(MODEL_PATH)
metrics = model.val(
    data=YAML_PATH,
    split='test',       # test셋으로 평가
    imgsz=640,
    batch=32,
    device=0,
    save_json=True,     # COCO 형식 결과 저장
    plots=True,         # 시각화 자동 저장
    project=RESULT_DIR,
    name="det_val",
)

# =============================================
# 결과 출력 및 저장
# =============================================
map50   = metrics.box.map50
map5095 = metrics.box.map
precision = metrics.box.mp
recall    = metrics.box.mr

print(f"[mAP50]     {map50:.4f}")
print(f"[mAP50-95]  {map5095:.4f}")
print(f"[Precision] {precision:.4f}")
print(f"[Recall]    {recall:.4f}")

with open(os.path.join(RESULT_DIR, "val_result.txt"), 'w') as f:
    f.write(f"mAP50:      {map50:.4f}\n")
    f.write(f"mAP50-95:   {map5095:.4f}\n")
    f.write(f"Precision:  {precision:.4f}\n")
    f.write(f"Recall:     {recall:.4f}\n")

end_time = datetime.now()
elapsed  = (end_time - start_time).total_seconds() / 60
print(f"[종료] {end_time.strftime('%Y-%m-%d %H:%M:%S')} / 소요: {elapsed:.1f}분")
print(f"[결과 저장] {RESULT_DIR}")
