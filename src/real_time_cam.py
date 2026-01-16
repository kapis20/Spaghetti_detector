import cv2
from ultralytics import YOLO
from pathlib import Path

# Repo root = parent of /src
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "spaghetti_best.pt"
model = YOLO(str(MODEL_PATH))
# Start capturing video from the default camera (0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Cannot open the camera.")
    exit()

# Process the video frame by frame
while True:
    # Capture each frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame.")
        break

    # Perform inference on the current frame
    results = model(frame)

    # Render the results on the frame
    annotated_frame = results[0].plot()  # Draw detections

    # Display the frame with detections
    cv2.imshow('YOLO Detection', annotated_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
