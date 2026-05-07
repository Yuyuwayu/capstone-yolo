"""
FishWatch — Fish Detector

YOLO-based fish detection with hunger analysis via average inter-fish distance.
Consolidates logic previously duplicated across main.py, main1.py, final.py, final2.py.
"""

import os
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from . import config


class FishDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or config.DEFAULT_MODEL_PATH
        self.model = None
        self.distance_threshold = config.DISTANCE_THRESHOLD
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        self.history = deque(maxlen=config.HISTORY_LENGTH)

        # Latest state (thread-safe via _lock)
        self.latest_frame = None
        self.latest_status = "Unknown"
        self.latest_avg_distance = 0.0
        self.latest_fish_count = 0
        self.latest_timestamp = ""

        self._lock = threading.Lock()
        self.cap = None
        self.source = None
        self._running = False
        self._thread = None

        self._load_model()

    # ── Model Management ──────────────────────────────

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"[FishDetector] Failed to load model: {e}")
            self.model = None

    def switch_model(self, model_path):
        """Hot-swap the YOLO model without restarting the stream."""
        with self._lock:
            self.model_path = model_path
            self._load_model()
            self.history.clear()
        return self.model is not None

    # ── Frame Processing ──────────────────────────────

    def process_frame(self, frame):
        """Run YOLO on a single frame, compute distances, return results."""
        if self.model is None:
            return None

        results = self.model(frame, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        centroids = []
        for i, box in enumerate(boxes):
            if confs[i] < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            centroids.append((cx, cy))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (178, 186, 60), 2)

        avg_dist = self._avg_distance(centroids)
        status = "Lapar" if avg_dist < self.distance_threshold else "Tidak Lapar"
        self.history.append(status)

        return {
            "frame": frame,
            "avg_distance": round(avg_dist, 2),
            "status": status,
            "smoothed_status": self._smoothed_status(),
            "fish_count": len(centroids),
        }

    @staticmethod
    def _avg_distance(centroids):
        n = len(centroids)
        if n < 2:
            return 9999.0
        dists = [
            np.linalg.norm(np.array(centroids[i]) - np.array(centroids[j]))
            for i in range(n)
            for j in range(i + 1, n)
        ]
        return float(np.mean(dists))

    def _smoothed_status(self):
        if not self.history:
            return "Unknown"
        return "Lapar" if self.history.count("Lapar") > len(self.history) // 2 else "Tidak Lapar"

    # ── Video Stream ──────────────────────────────────

    def start_stream(self, source=None):
        """Start background thread that reads frames and runs detection."""
        if self._running:
            self.stop_stream()

        src = config.WEBCAM_INDEX if (source is None or source == "webcam") else source

        # Numeric string → int (webcam index)
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        # Plain filename (no path sep) → resolve from videos/ dir
        if isinstance(src, str) and os.sep not in src and "/" not in src and not src.startswith("http"):
            candidate = os.path.join(config.VIDEOS_DIR, src)
            if os.path.isfile(candidate):
                src = candidate

        self.source = src
        self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            return False

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def _loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 1 / 30

        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str):  # video file → loop
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            result = self.process_frame(frame)
            if result:
                with self._lock:
                    self.latest_frame = result["frame"].copy()
                    self.latest_status = result["smoothed_status"]
                    self.latest_avg_distance = result["avg_distance"]
                    self.latest_fish_count = result["fish_count"]
                    self.latest_timestamp = datetime.now().strftime("%H:%M:%S")

            time.sleep(delay)

    def stop_stream(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self.cap:
            self.cap.release()
            self.cap = None

    # ── Accessors (thread-safe) ───────────────────────

    def get_latest(self):
        with self._lock:
            return {
                "status": self.latest_status,
                "avg_distance": self.latest_avg_distance,
                "fish_count": self.latest_fish_count,
                "timestamp": self.latest_timestamp,
                "has_frame": self.latest_frame is not None,
            }

    def get_frame_bytes(self):
        with self._lock:
            if self.latest_frame is None:
                return None
            _, buf = cv2.imencode(".jpg", self.latest_frame)
            return buf.tobytes()
