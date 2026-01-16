from ultralytics import YOLO
from pathlib import Path

# Repo root = parent of /src
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "spaghetti_best.pt"
DATA_PATH = ROOT / "data" / "data.yaml"

#model = YOLO("yolov8n.yaml")  # build a new model from scratch

# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# #epochs based on the dataset size
# # Train the model
# results = model.train(
#     data='data.yaml',
#     epochs=50,       # Start with 50 epochs
#     imgsz=640,       # Image size
#     batch=8,         # Batch size (adjust based on GPU RAM)
#     patience=10,     # Early stopping after 10 epochs with no improvement
#     workers=4        # Number of CPU workers for loading data
# )

# Evaluate the trained model on the validation set
model = YOLO(str(MODEL_PATH))
metrics = model.val(data=str(DATA_PATH)) # Evaluate on validation data

# Print evaluation results
print(metrics)
print(f"mAP50-95: {metrics.box.map:.3f}")      # mAP50-95
print(f"mAP50: {metrics.box.map50:.3f}")       # mAP50
print(f"Precision: {metrics.box.precision:.3f}")  # Precision
print(f"Recall: {metrics.box.recall:.3f}")        # Recall

 