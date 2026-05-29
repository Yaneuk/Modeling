import os
import shutil
from datetime import datetime
from ultralytics import YOLO


# =============================================
# 경로 설정 (RunPod 환경 기준)
# =============================================
DATASET_PATH = "/workspace/dataset/classification"
RESULT_PATH  = "/workspace/results/classification"

# .ipynb_checkpoints 제거 (주피터 환경 잔여물)
for root, dirs, _ in os.walk(DATASET_PATH):
    if '.ipynb_checkpoints' in dirs:
        shutil.rmtree(os.path.join(root, '.ipynb_checkpoints'))

os.makedirs(RESULT_PATH, exist_ok=True)

# =============================================
# 학습 시작
# =============================================
start_time = datetime.now()
print(f"[시작] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

model = YOLO("yolov8s-cls.pt")  # n→s로 업그레이드 (성능 향상)

results = model.train(
    data=DATASET_PATH,
    project=RESULT_PATH,
    name="cls_v1",
    epochs=50,        # 기존 20 → 100 (26만 장 규모에 맞게 조정)
    imgsz=640,
    patience=15,       # 기존 5 → 15 (조기종료 조건 완화)
    batch=16,          # 기존 100 → 16 (RunPod 16GB 기준 안정적)
    workers=4,
    cos_lr=True,       # 학습률 점진적 감소
    save=True,
    save_period=10,    # 10 epoch마다 중간 저장
    device=0,
)

# =============================================
# 결과 저장
# =============================================
end_time = datetime.now()
elapsed  = (end_time - start_time).total_seconds() / 60

print(f"[종료] {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"[소요] {elapsed:.1f}분")

accuracy = results.results_dict.get('metrics/accuracy_top1', None)
print(f"[Top1 Accuracy] {accuracy}")

os.makedirs(f"{RESULT_PATH}/cls_v1", exist_ok=True)
with open(f"{RESULT_PATH}/cls_v1/metrics.txt", 'w') as f:
    f.write(f"학습 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"학습 종료: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"소요 시간: {elapsed:.1f}분\n")
    f.write(f"Top1 Accuracy: {accuracy}\n")

print(f"[완료] 모델 저장 위치: {RESULT_PATH}/cls_v1/weights/best.pt")
