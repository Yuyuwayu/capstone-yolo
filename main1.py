from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import time
from threading import Thread, Lock
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USE_VIDEO = True
VIDEO_PATH = "C:/Users/justt/Downloads/1.mp4"
WEBCAM_INDEX = 1

model = YOLO('runs/detect/train10/weights/best.pt')

cap = cv2.VideoCapture(VIDEO_PATH) if USE_VIDEO else cv2.VideoCapture(WEBCAM_INDEX)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = 1 / fps if fps else 1 / 30

frame_lock = Lock()
latest_frame = None
detection_status = "Unknown"
average_dist = 0.0
fish_count = 0

def average_distance(centroids):
    n = len(centroids)
    if n < 2:
        return 9999
    dists = [
        np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
        for i in range(n) for j in range(i + 1, n)
    ]
    return np.mean(dists)

def processing_thread():
    global latest_frame, detection_status, average_dist, fish_count

    while True:
        ret, frame = cap.read()
        if not ret:
            if USE_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
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
            fish_count = len(centroids)

        time.sleep(delay)

Thread(target=processing_thread, daemon=True).start()

@app.get("/deteksi")
def deteksi():
    with frame_lock:
        if latest_frame is None:
            return {"status": "error", "message": "Belum ada frame"}

        _, buffer = cv2.imencode('.jpg', latest_frame)
        img_str = base64.b64encode(buffer).decode('utf-8')

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return {
            "status": detection_status,
            "avg_distance": average_dist,
            "count": fish_count,
            "image": img_str,
            "timestamp": timestamp
        }

@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                _, buffer = cv2.imencode('.jpg', latest_frame)
                frame_bytes = buffer.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpg\r\n\r\n' + frame_bytes + b'\r\n'
            )
            time.sleep(delay)

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')