"""
FishWatch — Centralized Configuration

All hardcoded values live here. Can be updated at runtime via API.
"""

import os


def _env_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default

# ── Paths ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

# ── Model Defaults ────────────────────────────────────
DEFAULT_MODEL_RUN = "train18"
DEFAULT_MODEL_PATH = os.path.join(RUNS_DIR, DEFAULT_MODEL_RUN, "weights", "best.pt")

# ── Detection ─────────────────────────────────────────
DISTANCE_THRESHOLD = 300
CONFIDENCE_THRESHOLD = _env_float("FISHWATCH_CONFIDENCE_THRESHOLD", 0.25)
HISTORY_LENGTH = 30
SMOOTHING_WINDOW_SECONDS = 30  # Average distances over this time window (seconds)
INFERENCE_IMAGE_SIZE = _env_int("FISHWATCH_INFERENCE_IMAGE_SIZE", 320)
PROCESS_EVERY_N_FRAMES = max(1, _env_int("FISHWATCH_PROCESS_EVERY_N_FRAMES", 3))
STREAM_JPEG_QUALITY = max(30, min(95, _env_int("FISHWATCH_STREAM_JPEG_QUALITY", 65)))

# ── Video Source ──────────────────────────────────────
WEBCAM_INDEX = 0
DEFAULT_SOURCE = "webcam"
CAMERA_WIDTH = _env_int("FISHWATCH_CAMERA_WIDTH", 640)
CAMERA_HEIGHT = _env_int("FISHWATCH_CAMERA_HEIGHT", 480)
CAMERA_FPS = _env_int("FISHWATCH_CAMERA_FPS", 10)

# ── Dataset Layouts ───────────────────────────────────
# Datasets are now dynamically discovered from the DATASET_DIR.

# ── Training Defaults ─────────────────────────────────
DEFAULT_BASE_MODEL = "yolov8n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 16
DEFAULT_IMG_SIZE = 640

# ── Environment Check ────────────────────────────────
REQUIRED_PACKAGES = {
    "ultralytics": "ultralytics",
    "cv2": "opencv-python",
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "scipy": "scipy",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
}

REQUIRED_DIRECTORIES = [RUNS_DIR, DATASET_DIR, VIDEOS_DIR]
