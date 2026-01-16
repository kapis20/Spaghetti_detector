# Spaghetti Detection for 3D Printing (YOLO)

Detect **“spaghetti” 3D print failures** using a YOLO object detector trained on a **public labelled dataset** (YOLO-format labels).  
This repo includes training outputs, exported model files, and simple scripts to run **evaluation** and **real-time webcam detection**.

🎥 **Demo video:** https://www.youtube.com/watch?v=-KYh9fgHfVY

---

## What this project does

- Trains a YOLO model to detect spaghetti-like print failures.
- Evaluates the trained model and prints metrics (mAP, precision, recall).
- Runs inference on a **live camera feed** and draws bounding boxes in real time.

---

## Repository structure

```text
.
├─ data/
│  ├─ data.yaml
│  └─ dataset/
│     ├─ train/
│     ├─ val/
│     └─ test/
├─ models/
│  ├─ spaghetti_best.pt          # recommended: main trained weights for inference
│  ├─ yolov8n.pt                 # baseline/pretrained weights (optional)
│  └─ spaghetti.tflite / yolov8.tflite (optional)
├─ src/
│  ├─ spaghetti_detection.py     # evaluate model on validation set
│  ├─ real_time_cam.py           # real-time webcam inference
│  ├─ export/                    # conversion/export scripts
│  └─ utils/                     # helpers (e.g., dataset conversion)
├─ scripts/                      # misc test scripts
├─ runs/                         # Ultralytics training outputs (generated)
└─ artifacts/                    # extra files/exports/legacy items

## Contributors
- Kacper Sikorski (@kapis20)