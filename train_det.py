import os
import shutil
from datetime import datetime
from ultralytics import YOLO


# =============================================
# 경로 설정 (RunPod 환경 기준)
# =============================================
YAML_PATH   = "/workspace/Modeling/data_det.yaml"
RESULT_PATH = "/workspace/results/detection"

# .ipynb_checkpoints 제거
for root, dirs, _ in os.walk('/workspace/dataset/detection'):
    if '.ipynb_checkpoints' in dirs:
        shutil.rmtree(os.path.join(root, '.ipynb_checkpoints'))

os.makedirs(RESULT_PATH, exist_ok=True)

# =============================================
# 학습 시작
# =============================================
start_time = datetime.now()
print(f"[시작] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

model = YOLO("yolov8s.pt")  # 탐지 모델 (cls 아님)

results = model.train(
    data=YAML_PATH,
    project=RESULT_PATH,
    name="det_v1",
    epochs=50,
    imgsz=640,
    patience=15,
    batch=16,
    workers=4,
    cos_lr=True,
    save=True,
    save_period=10,
    device=0,

    # 클래스 불균형 대응 (dispose가 10%로 적으므로)
    cls=1.5,           # 클래스 손실 가중치 높임
)

# =============================================
# 결과 저장
# =============================================
end_time = datetime.now()
elapsed  = (end_time - start_time).total_seconds() / 60

print(f"[종료] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[소요] {elapsed:.1f}분")

map50   = results.results_dict.get('metrics/mAP50(B)', None)
map5095 = results.results_dict.get('metrics/mAP50-95(B)', None)
print(f"[mAP50]    {map50}")
print(f"[mAP50-95] {map5095}")

os.makedirs(f"{RESULT_PATH}/det_v1", exist_ok=True)
with open(f"{RESULT_PATH}/det_v1/metrics.txt", 'w') as f:
    f.write(f"학습 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"학습 종료: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"소요 시간: {elapsed:.1f}분\n")
    f.write(f"mAP50:     {map50}\n")
    f.write(f"mAP50-95:  {map5095}\n")

print(f"[완료] 모델 저장 위치: {RESULT_PATH}/det_v1/weights/best.pt")
