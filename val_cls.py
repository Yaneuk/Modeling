import os
import shutil
from datetime import datetime
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================
# 경로 설정
# =============================================
MODEL_PATH  = "/workspace/results/classification/cls_v1/weights/best.pt"
TEST_DIR    = "/workspace/dataset/classification/test"
RESULT_DIR  = "/workspace/results/classification/cls_v1/val"
OUTPUT_TXT  = os.path.join(RESULT_DIR, "classification_results.txt")
METRIC_TXT  = os.path.join(RESULT_DIR, "val_result.txt")

os.makedirs(RESULT_DIR, exist_ok=True)

# .ipynb_checkpoints 제거
for root, dirs, _ in os.walk(TEST_DIR):
    if '.ipynb_checkpoints' in dirs:
        shutil.rmtree(os.path.join(root, '.ipynb_checkpoints'))

# =============================================
# 예측 수행
# =============================================
start_time = datetime.now()
print(f"[시작] {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

model = YOLO(MODEL_PATH)

with open(OUTPUT_TXT, 'w') as f:
    for class_name in os.listdir(TEST_DIR):
        class_folder = os.path.join(TEST_DIR, class_name)
        if not os.path.isdir(class_folder):
            continue

        # 배치 처리 (기존 1장씩 → 폴더 단위로 묶어서 처리)
        image_paths = [
            os.path.join(class_folder, fn)
            for fn in os.listdir(class_folder)
            if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
        ]

        if not image_paths:
            continue

        # 배치 단위로 예측 (속도 대폭 향상)
        batch_size = 32
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            results = model.predict(batch, verbose=False)

            for img_path, result in zip(batch, results):
                filename  = os.path.basename(img_path)
                top1_cls  = result.probs.top1
                top1_conf = float(result.probs.top1conf)
                f.write(
                    f"File: {filename}, "
                    f"Predicted Class: {top1_cls}, "
                    f"Actual Class: {class_name}, "
                    f"Confidence: {top1_conf:.4f}\n"
                )

print("[예측 완료]")

# =============================================
# 성능 지표 계산
# =============================================
true_labels = []
pred_labels = []

with open(OUTPUT_TXT, 'r') as f:
    for line in f:
        parts      = line.strip().split(', ')
        true_cls   = parts[2].split(': ')[1].strip()
        pred_idx   = int(parts[1].split(': ')[1].strip())
        pred_cls   = model.names[pred_idx]
        true_labels.append(true_cls)
        pred_labels.append(pred_cls)

accuracy = accuracy_score(true_labels, pred_labels)
labels   = unique_labels(true_labels, pred_labels)
f1       = f1_score(true_labels, pred_labels, average=None, labels=labels)
f1_macro = f1_score(true_labels, pred_labels, average='macro')
f1_weighted = f1_score(true_labels, pred_labels, average='weighted')

print(f"[Accuracy]       {accuracy:.4f}")
print(f"[F1 macro]       {f1_macro:.4f}")
print(f"[F1 weighted]    {f1_weighted:.4f}")

with open(METRIC_TXT, 'w') as f:
    f.write(f"Accuracy:         {accuracy:.4f}\n")
    for label, score in zip(labels, f1):
        f.write(f"F1 [{label:>10}]: {score:.5f}\n")
    f.write(f"F1 macro:         {f1_macro:.8f}\n")
    f.write(f"F1 weighted:      {f1_weighted:.8f}\n")

# =============================================
# Confusion Matrix 저장
# =============================================
cm = confusion_matrix(true_labels, pred_labels, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix — Classification')
plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()

end_time = datetime.now()
elapsed  = (end_time - start_time).total_seconds() / 60
print(f"[종료] {end_time.strftime('%Y-%m-%d %H:%M:%S')} / 소요: {elapsed:.1f}분")
print(f"[결과 저장] {RESULT_DIR}")
