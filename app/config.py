"""
FishWatch — Centralized Configuration

All hardcoded values live here. Can be updated at runtime via API.
"""

import os

# ── Paths ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")

# ── Model Defaults ────────────────────────────────────
DEFAULT_MODEL_RUN = "train10"
DEFAULT_MODEL_PATH = os.path.join(RUNS_DIR, DEFAULT_MODEL_RUN, "weights", "best.pt")

# ── Detection ─────────────────────────────────────────
DISTANCE_THRESHOLD = 300
CONFIDENCE_THRESHOLD = 0.5
HISTORY_LENGTH = 30

# ── Video Source ──────────────────────────────────────
WEBCAM_INDEX = 1
DEFAULT_SOURCE = "webcam"

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
