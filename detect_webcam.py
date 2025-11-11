# detect_webcam.py
import cv2
from ultralytics import YOLO
import time

# Charger le modèle YOLOv8 nano (léger)
model = YOLO("yolov8n.pt")

# Ouvrir la webcam (0 = caméra par défaut)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur: impossible d'accéder à la webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()
    results = model(frame, conf=0.4)
    annotated = results[0].plot()
    fps = 1.0 / max(time.time() - t0, 1e-6)

    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
