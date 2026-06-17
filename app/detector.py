"""
FishWatch — Fish Detector

YOLO-based fish detection with hunger analysis via time-windowed average
inter-fish distance. Status is determined by averaging distances over a
configurable time window (default 30 seconds) for stable, flicker-free results.
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
        self.smoothing_window = config.SMOOTHING_WINDOW_SECONDS
        self.inference_image_size = config.INFERENCE_IMAGE_SIZE
        self.process_every_n_frames = config.PROCESS_EVERY_N_FRAMES
        self.stream_jpeg_quality = config.STREAM_JPEG_QUALITY

        # Time-windowed history: stores (timestamp_float, avg_distance) tuples
        self._distance_history = deque(maxlen=1800)  # ~60s at 30fps

        # Latest state (thread-safe via _lock)
        self.latest_frame = None
        self.latest_status = "Unknown"
        self.latest_avg_distance = 0.0
        self.latest_windowed_avg = 0.0
        self.latest_fish_count = 0
        self.latest_timestamp = ""

        self._lock = threading.Lock()
        self.cap = None
        self.source = None
        self._running = False
        self._thread = None
        self._frame_counter = 0

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
            self._distance_history.clear()
        return self.model is not None

    # ── Frame Processing ──────────────────────────────

    def process_frame(self, frame):
        """Run YOLO on a single frame, compute distances, return results."""
        if self.model is None:
            return {
                "frame": frame,
                "avg_distance": 0.0,
                "windowed_avg": 0.0,
                "status": "No model",
                "smoothed_status": "No model",
                "fish_count": 0,
            }

        results = self.model(frame, imgsz=self.inference_image_size, verbose=False)[0]
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
        now = time.time()

        # Store this frame's distance with timestamp
        self._distance_history.append((now, avg_dist))

        # Compute windowed average
        windowed_avg = self._windowed_average(now)
        status = "Lapar" if windowed_avg < self.distance_threshold else "Tidak Lapar"

        return {
            "frame": frame,
            "avg_distance": round(avg_dist, 2),
            "windowed_avg": round(windowed_avg, 2),
            "status": status,
            "smoothed_status": status,
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

    def _windowed_average(self, now):
        """Compute average distance over the smoothing time window."""
        cutoff = now - self.smoothing_window
        # Collect distances within the time window
        window_dists = [d for t, d in self._distance_history if t >= cutoff]
        if not window_dists:
            return 9999.0
        return float(np.mean(window_dists))

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
        self.cap = self._open_capture(src)

        if not self.cap.isOpened():
            return False

        self._configure_capture(src)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    @staticmethod
    def _open_capture(src):
        """Open camera/video source with Windows-friendly webcam fallback."""
        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(src)

    def _configure_capture(self, src):
        """Request low camera settings when using webcam-like sources."""
        if not isinstance(src, int) or self.cap is None:
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)

    def _loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / fps if fps > 0 else 1 / 30

        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str):  # video file → loop
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self._frame_counter += 1
            should_process = (
                self.latest_frame is None
                or self.process_every_n_frames <= 1
                or self._frame_counter % self.process_every_n_frames == 0
            )
            if not should_process:
                time.sleep(delay)
                continue

            result = self.process_frame(frame)
            if result:
                with self._lock:
                    self.latest_frame = result["frame"].copy()
                    self.latest_status = result["smoothed_status"]
                    self.latest_avg_distance = result["avg_distance"]
                    self.latest_windowed_avg = result["windowed_avg"]
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
                "windowed_avg": self.latest_windowed_avg,
                "fish_count": self.latest_fish_count,
                "timestamp": self.latest_timestamp,
                "has_frame": self.latest_frame is not None,
            }

    def get_frame_bytes(self):
        with self._lock:
            if self.latest_frame is None:
                return None
            _, buf = cv2.imencode(
                ".jpg",
                self.latest_frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.stream_jpeg_quality],
            )
            return buf.tobytes()
