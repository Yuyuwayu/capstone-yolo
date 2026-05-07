import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import pdist
from collections import deque

# ==== Konfigurasi ====
model_path = "runs/detect/train/weights/best.pt"  # model YOLOv8 yang sudah dilatih
# ip = "192.168.1.7"
# port = 4747
distance_threshold = 300
conf_threshold = 0.5
history_length = 10

# ==== Load Model ====
model = YOLO(model_path)
# cap = cv2.VideoCapture(f"http://{ip}:{port}/video")  # stream DroidCam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Gagal membuka stream dari DroidCam. Pastikan HP dan PC 1 jaringan dan DroidCam berjalan.")

status_history = deque(maxlen=history_length)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Tidak bisa membaca frame dari stream.")
        break

    results = model(frame)[0]
    centers = []

    for box in results.boxes:
        if box.conf < conf_threshold:
            continue
        cls = int(box.cls)
        if cls != 0:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        centers.append([cx, cy])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    current_status = "Tidak Lapar"
    if len(centers) >= 2:
        dists = pdist(centers)
        avg_dist = np.mean(dists)

        if avg_dist < distance_threshold:
            current_status = "LAPAR"

        cv2.putText(frame, f"Avg Dist: {avg_dist:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    status_history.append(current_status)
    if status_history.count("LAPAR") > len(status_history) // 2:
        status = "LAPAR"
    else:
        status = "Tidak Lapar"

    color = (0, 0, 255) if status == "LAPAR" else (0, 255, 0)
    cv2.putText(frame, f"Ikan {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    cv2.imshow("Deteksi Nafsu Makan Ikan (DroidCam)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
