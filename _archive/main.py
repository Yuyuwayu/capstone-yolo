from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import time
from threading import Thread, Lock
import queue
from datetime import datetime
from zoneinfo import ZoneInfo  # jika belum, tambahkan

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model sekali saja
model = YOLO("runs/detect/train3/weights/best.pt")
cap = cv2.VideoCapture(1)  # webcam
#cap = cv2.VideoCapture("http://192.168.1.101:81/stream")

# Variabel global dan lock
frame_lock = Lock()
latest_frame = None
detection_status = "Unknown"
average_dist = 0.0

# Fungsi bantu: hitung rata-rata jarak antar centroid
def average_distance(centroids):
    n = len(centroids)
    if n < 2:
        return 9999
    dists = [
        np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
        for i in range(n) for j in range(i + 1, n)
    ]
    return np.mean(dists)

# Thread: proses frame dari webcam dan deteksi YOLO
def processing_thread():
    global latest_frame, detection_status, average_dist
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame)[0]
        boxes = results.boxes.xyxy.cpu().numpy()

        centroids = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centroids.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        avg_dist = average_distance(centroids)
        status = "Lapar" if avg_dist < 300 else "Tidak Lapar"

        with frame_lock:
            latest_frame = frame.copy()
            detection_status = status
            average_dist = round(avg_dist, 2)

        time.sleep(0.03)  # ~30 FPS

# Mulai thread background saat server dijalankan
Thread(target=processing_thread, daemon=True).start()

# Endpoint untuk API status dan gambar (base64)
@app.get("/deteksi")
def deteksi():
    with frame_lock:
        if latest_frame is None:
            return {"status": "error", "message": "Belum ada frame"}

        _, buffer = cv2.imencode('.jpg', latest_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')

        # Waktu lokal Jakarta (UTC+7)
        timestamp_wib = datetime.now(ZoneInfo("Asia/Jakarta")).strftime('%Y-%m-%d %H:%M:%S')

        return {
            "status": detection_status,
            "avg_distance": average_dist,
            "image": img_str,
            "timestamp": timestamp_wib
        }


# Endpoint streaming video langsung
@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                _, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')
