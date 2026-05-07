"""
FishWatch — FastAPI Server

Single entry point that wires up all modules and exposes API endpoints.
Run with: uvicorn app.server:app --host 0.0.0.0 --port 8000
"""

import base64
import os
import time

import cv2
import numpy as np
from fastapi import FastAPI, Query, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List

from . import config
from .env_checker import EnvChecker
from .model_manager import ModelManager
from .dataset_manager import DatasetManager
from .trainer import YOLOTrainer

# ── App Setup ─────────────────────────────────────────

app = FastAPI(title="FishWatch", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons
env_checker = EnvChecker()
model_manager = ModelManager()
dataset_manager = DatasetManager()
trainer = YOLOTrainer()
detector = None  # lazy-loaded after env check

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _get_detector():
    """Lazy-load detector so the app starts even without torch."""
    global detector
    if detector is None:
        try:
            from .detector import FishDetector
            detector = FishDetector(model_manager.get_active_path())
        except Exception as e:
            print(f"[server] Could not init detector: {e}")
            return None
    return detector


# ── Dashboard ─────────────────────────────────────────

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ══════════════════════════════════════════════════════
# ENVIRONMENT API
# ══════════════════════════════════════════════════════

@app.get("/api/env/check")
def env_check():
    return env_checker.check_all()


@app.post("/api/env/install")
def env_install():
    return env_checker.install_missing()


@app.get("/api/env/install/status")
def env_install_status():
    return env_checker.get_install_status()

# ══════════════════════════════════════════════════════
# MONITOR API
# ══════════════════════════════════════════════════════

detection_history = []


@app.get("/api/monitor/status")
def monitor_status():
    det = _get_detector()
    if det is None:
        return {"status": "No detector", "avg_distance": 0, "fish_count": 0, "timestamp": "", "has_frame": False}
    data = det.get_latest()
    # Append to history
    if data["has_frame"]:
        detection_history.append({
            "status": data["status"],
            "avg_distance": data["avg_distance"],
            "fish_count": data["fish_count"],
            "timestamp": data["timestamp"],
        })
        if len(detection_history) > 200:
            detection_history.pop(0)
    return data


@app.get("/api/monitor/video_feed")
def monitor_video_feed():
    det = _get_detector()
    if det is None:
        raise HTTPException(503, "Detector not available.")

    def generate():
        while True:
            frame_bytes = det.get_frame_bytes()
            if frame_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.033)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/monitor/history")
def monitor_history():
    return detection_history[-100:]


class MonitorConfig(BaseModel):
    distance_threshold: Optional[float] = None
    confidence_threshold: Optional[float] = None
    source: Optional[str] = None


@app.get("/api/monitor/config")
def monitor_config_get():
    det = _get_detector()
    return {
        "distance_threshold": det.distance_threshold if det else config.DISTANCE_THRESHOLD,
        "confidence_threshold": det.confidence_threshold if det else config.CONFIDENCE_THRESHOLD,
        "source": str(det.source) if det else config.DEFAULT_SOURCE,
        "active_model": model_manager.active_model,
    }


@app.post("/api/monitor/config")
def monitor_config_set(cfg: MonitorConfig):
    det = _get_detector()
    if det is None:
        raise HTTPException(503, "Detector not available.")
    if cfg.distance_threshold is not None:
        det.distance_threshold = cfg.distance_threshold
    if cfg.confidence_threshold is not None:
        det.confidence_threshold = cfg.confidence_threshold
    if cfg.source is not None:
        det.stop_stream()
        det.start_stream(cfg.source)
    return {"success": True}


@app.post("/api/monitor/start")
def monitor_start(source: Optional[str] = None):
    det = _get_detector()
    if det is None:
        raise HTTPException(503, "Detector not available.")
    ok = det.start_stream(source)
    return {"success": ok}


@app.post("/api/monitor/stop")
def monitor_stop():
    det = _get_detector()
    if det:
        det.stop_stream()
    return {"success": True}


@app.get("/api/monitor/cameras")
def detect_cameras():
    """Auto-detect available webcam indices (probes 0-4)."""
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DSHOW is faster on Windows
        if cap.isOpened():
            cameras.append({"index": i, "name": f"Webcam {i}"})
            cap.release()
    return cameras


@app.get("/api/monitor/videos")
def list_videos():
    """List video files in the videos/ directory."""
    if not os.path.isdir(config.VIDEOS_DIR):
        return []
    exts = (".mp4", ".avi", ".mkv", ".mov", ".wmv")
    return sorted(f for f in os.listdir(config.VIDEOS_DIR) if f.lower().endswith(exts))


@app.post("/api/monitor/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file to the videos/ directory."""
    os.makedirs(config.VIDEOS_DIR, exist_ok=True)
    path = os.path.join(config.VIDEOS_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"success": True, "filename": file.filename}


@app.post("/api/monitor/process_frame")
async def process_frame(file: UploadFile = File(...)):
    """Process a single frame from the browser camera. Returns annotated image + data."""
    det = _get_detector()
    if det is None:
        raise HTTPException(503, "Detector not available.")

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image data.")

    from datetime import datetime
    result = det.process_frame(frame)
    if result is None:
        raise HTTPException(500, "Detection failed.")

    _, buf = cv2.imencode(".jpg", result["frame"], [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64 = base64.b64encode(buf.tobytes()).decode()

    ts = datetime.now().strftime("%H:%M:%S")
    detection_history.append({
        "status": result["smoothed_status"],
        "avg_distance": result["avg_distance"],
        "fish_count": result["fish_count"],
        "timestamp": ts,
    })
    if len(detection_history) > 200:
        detection_history.pop(0)

    return {
        "image": b64,
        "status": result["smoothed_status"],
        "avg_distance": result["avg_distance"],
        "fish_count": result["fish_count"],
        "timestamp": ts,
    }

# ══════════════════════════════════════════════════════
# DATASET API
# ══════════════════════════════════════════════════════

@app.get("/api/dataset/list")
def dataset_list_names():
    """Return all configured dataset names."""
    return list(config.DATASETS.keys())


@app.get("/api/dataset/stats")
def dataset_stats():
    return dataset_manager.scan_datasets()


class MergeRequest(BaseModel):
    sources: list[str]
    target: str = "merged"


@app.post("/api/dataset/merge")
def dataset_merge(req: MergeRequest):
    return dataset_manager.merge_datasets(req.sources, req.target)


@app.get("/api/dataset/{name}/images")
def dataset_images(name: str, split: str = "train", filter: str = "all"):
    return dataset_manager.list_images(name, split, filter)


@app.get("/api/dataset/{name}/image/{split}/{filename}")
def dataset_image(name: str, split: str, filename: str):
    path = dataset_manager.get_image_path(name, split, filename)
    if not path:
        raise HTTPException(404, "Image not found.")
    return FileResponse(path)


@app.get("/api/dataset/{name}/preview/{split}/{filename}")
def dataset_preview(name: str, split: str, filename: str):
    data = dataset_manager.get_annotated_preview(name, split, filename)
    if not data:
        raise HTTPException(404, "Image not found.")
    return Response(content=data, media_type="image/jpeg")


class ImportRequest(BaseModel):
    source_path: str
    dataset_name: str = "custom"


@app.post("/api/dataset/import")
def dataset_import(req: ImportRequest):
    return dataset_manager.import_folder(req.source_path, req.dataset_name)


@app.post("/api/dataset/{name}/upload-images")
async def dataset_upload_images(
    name: str,
    train_ratio: float = 0.8,
    files: list[UploadFile] = File(...)
):
    """Upload images and auto-split into train/val based on ratio."""
    import random

    ds = config.DATASETS.get(name)
    if not ds:
        raise HTTPException(404, f"Unknown dataset '{name}'.")

    # Get train and val paths
    train_split = ds["splits"].get("train")
    val_split = ds["splits"].get("val") or ds["splits"].get("valid")

    if not train_split:
        raise HTTPException(400, "Dataset has no train split configured.")

    train_img_dir = train_split["images"]
    os.makedirs(train_img_dir, exist_ok=True)

    val_img_dir = None
    if val_split:
        val_img_dir = val_split["images"]
        os.makedirs(val_img_dir, exist_ok=True)

    # Read all file data first
    file_data = []
    for f in files:
        if not f.filename:
            continue
        data = await f.read()
        file_data.append((f.filename, data))

    # Shuffle and split
    random.shuffle(file_data)
    split_idx = max(1, int(len(file_data) * train_ratio))
    train_files = file_data[:split_idx]
    val_files = file_data[split_idx:] if val_img_dir else []

    # Save train
    for fname, data in train_files:
        with open(os.path.join(train_img_dir, fname), "wb") as out:
            out.write(data)

    # Save val
    for fname, data in val_files:
        with open(os.path.join(val_img_dir, fname), "wb") as out:
            out.write(data)

    return {
        "success": True,
        "imported": len(file_data),
        "train": len(train_files),
        "val": len(val_files),
        "dataset": name,
    }


class SplitRequest(BaseModel):
    train_ratio: float = 0.8


@app.post("/api/dataset/{name}/split")
def dataset_split(name: str, req: SplitRequest):
    return dataset_manager.split_dataset(name, req.train_ratio)


@app.delete("/api/dataset/{name}/image/{split}/{filename}")
def dataset_delete_image(name: str, split: str, filename: str):
    return dataset_manager.delete_image(name, split, filename)


class BulkDeleteRequest(BaseModel):
    filenames: List[str]


@app.post("/api/dataset/{name}/delete-bulk/{split}")
def dataset_delete_bulk(name: str, split: str, req: BulkDeleteRequest):
    return dataset_manager.delete_images_bulk(name, split, req.filenames)

# ══════════════════════════════════════════════════════
# TRAINING API
# ══════════════════════════════════════════════════════

class TrainRequest(BaseModel):
    model: str = "yolov8n.pt"
    dataset: str = "roboflow"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: str = "cpu"


@app.post("/api/training/start")
def training_start(req: TrainRequest):
    # Auto-generate YAML from the selected dataset
    yaml_result = dataset_manager.generate_training_yaml(req.dataset)
    if not yaml_result.get("success"):
        return {"success": False, "error": yaml_result.get("error", "Failed to generate YAML.")}

    cfg = req.model_dump()
    cfg["data"] = yaml_result["yaml_file"]
    return trainer.start_training(cfg)


@app.get("/api/training/status")
def training_status():
    return trainer.get_status()


@app.get("/api/training/logs")
def training_logs():
    return trainer.get_full_logs()


@app.post("/api/training/stop")
def training_stop():
    return trainer.stop_training()


@app.get("/api/training/yamls")
def training_yamls():
    return trainer.list_data_yamls()


@app.get("/api/dataset/{name}/classes")
def dataset_classes(name: str):
    """Return detected class IDs and names from the dataset's labels."""
    names = dataset_manager.get_class_names(name)
    return {"classes": [{"id": k, "name": v} for k, v in names.items()]}


@app.post("/api/dataset/{name}/generate-yaml")
def dataset_generate_yaml(name: str):
    """Auto-generate a training YAML for this dataset."""
    return dataset_manager.generate_training_yaml(name)

# ══════════════════════════════════════════════════════
# MODELS API
# ══════════════════════════════════════════════════════

@app.get("/api/models")
def models_list():
    return model_manager.list_models()


@app.get("/api/models/{name}/curves")
def models_curves(name: str):
    data = model_manager.get_curves(name)
    if data is None:
        raise HTTPException(404, "No training curves found.")
    return data


@app.post("/api/models/{name}/activate")
def models_activate(name: str):
    ok = model_manager.set_active(name)
    if not ok:
        raise HTTPException(404, "Model weights not found.")
    # Hot-swap detector model
    det = _get_detector()
    if det:
        det.switch_model(model_manager.get_active_path())
    return {"success": True, "active": name}


@app.delete("/api/models/{name}")
def models_delete(name: str):
    result = model_manager.delete_model(name)
    if not result["success"]:
        raise HTTPException(400, result["error"])
    return result
