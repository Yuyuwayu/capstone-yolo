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
        return {"status": "No detector", "avg_distance": 0, "windowed_avg": 0, "fish_count": 0, "timestamp": "", "has_frame": False}
    data = det.get_latest()
    # Append to history
    if data["has_frame"]:
        detection_history.append({
            "status": data["status"],
            "avg_distance": data["avg_distance"],
            "windowed_avg": data["windowed_avg"],
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
    smoothing_window: Optional[int] = None
    source: Optional[str] = None


@app.get("/api/monitor/config")
def monitor_config_get():
    det = _get_detector()
    return {
        "distance_threshold": det.distance_threshold if det else config.DISTANCE_THRESHOLD,
        "confidence_threshold": det.confidence_threshold if det else config.CONFIDENCE_THRESHOLD,
        "smoothing_window": det.smoothing_window if det else config.SMOOTHING_WINDOW_SECONDS,
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
    if cfg.smoothing_window is not None:
        det.smoothing_window = cfg.smoothing_window
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
        ok = cap.isOpened()
        ret, _frame = cap.read() if ok else (False, None)
        if ok and ret:
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

    _, buf = cv2.imencode(
        ".jpg",
        result["frame"],
        [cv2.IMWRITE_JPEG_QUALITY, config.STREAM_JPEG_QUALITY],
    )
    b64 = base64.b64encode(buf.tobytes()).decode()

    ts = datetime.now().strftime("%H:%M:%S")
    detection_history.append({
        "status": result["smoothed_status"],
        "avg_distance": result["avg_distance"],
        "windowed_avg": result["windowed_avg"],
        "fish_count": result["fish_count"],
        "timestamp": ts,
    })
    if len(detection_history) > 200:
        detection_history.pop(0)

    return {
        "image": b64,
        "status": result["smoothed_status"],
        "avg_distance": result["avg_distance"],
        "windowed_avg": result["windowed_avg"],
        "fish_count": result["fish_count"],
        "timestamp": ts,
    }

# ══════════════════════════════════════════════════════
# DATASET API
# ══════════════════════════════════════════════════════

@app.get("/api/dataset/list")
def dataset_list_names():
    """Return all configured dataset names."""
    if not os.path.isdir(config.DATASET_DIR):
        return []
    return [d for d in os.listdir(config.DATASET_DIR) if os.path.isdir(os.path.join(config.DATASET_DIR, d))]

@app.post("/api/dataset/{name}/create")
def dataset_create(name: str):
    res = dataset_manager.create_dataset(name)
    if not res.get("success"):
        raise HTTPException(400, res.get("error", "Failed to create dataset."))
    return res

@app.delete("/api/dataset/{name}")
def dataset_delete(name: str):
    res = dataset_manager.delete_dataset(name)
    if not res.get("success"):
        raise HTTPException(400, res.get("error", "Failed to delete dataset."))
    return res


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
    """Upload images (and optionally labels) and auto-split into train/val based on ratio."""
    import random
    import os

    train_img_dir, train_lbl_dir = dataset_manager._dirs(name, "train")
    val_img_dir, val_lbl_dir = dataset_manager._dirs(name, "val")
    
    if not train_img_dir:
        raise HTTPException(404, f"Dataset '{name}' not found.")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Collect existing filenames to skip duplicates
    existing_train = set(os.listdir(train_img_dir))
    existing_val = set(os.listdir(val_img_dir))

    images_data = {}
    labels_data = {}
    pre_split = {}  # stem -> "train" or "val"

    # Group files by stem
    for f in files:
        if not f.filename:
            continue
        data = await f.read()
        
        # Parse path — use forward slashes consistently
        safe_path = f.filename.replace('\\', '/')
        parts = safe_path.split('/')
        
        # Determine split from folder names (case-insensitive)
        assigned_split = None
        for p in parts[:-1]:  # exclude filename itself
            p_lower = p.lower()
            if p_lower in ("val", "valid", "validation", "test"):
                assigned_split = "val"
                break
            elif p_lower in ("train", "training"):
                assigned_split = "train"
                break
                
        # Keep original case for the actual filename
        basename = parts[-1]
        ext = os.path.splitext(basename)[1].lower()
        stem = os.path.splitext(basename)[0]
        
        if assigned_split:
            pre_split[stem] = assigned_split
            
        if ext == ".txt":
            labels_data[stem] = data
        else:
            images_data[stem] = (basename, data)

    # Shuffle and split image stems
    stems = list(images_data.keys())
    if not stems:
        return {"success": False, "error": "No valid images found to upload."}
        
    random.shuffle(stems)
    
    # Process splits
    train_stems = []
    val_stems = []
    
    if pre_split:
        # Use folder-based split detection
        for stem in stems:
            split_choice = pre_split.get(stem, "train")
            if split_choice == "val":
                val_stems.append(stem)
            else:
                train_stems.append(stem)
    else:
        # Fallback to ratio-based split
        split_idx = max(1, int(len(stems) * train_ratio))
        train_stems = stems[:split_idx]
        val_stems = stems[split_idx:]

    skipped = 0

    # Save train
    for stem in train_stems:
        fname, data = images_data[stem]
        if fname in existing_train:
            skipped += 1
            continue
        with open(os.path.join(train_img_dir, fname), "wb") as out:
            out.write(data)
        if stem in labels_data:
            with open(os.path.join(train_lbl_dir, stem + ".txt"), "wb") as out:
                out.write(labels_data[stem])

    # Save val
    for stem in val_stems:
        fname, data = images_data[stem]
        if fname in existing_val:
            skipped += 1
            continue
        with open(os.path.join(val_img_dir, fname), "wb") as out:
            out.write(data)
        if stem in labels_data:
            with open(os.path.join(val_lbl_dir, stem + ".txt"), "wb") as out:
                out.write(labels_data[stem])

    imported = len(stems) - skipped
    return {
        "success": True,
        "imported": imported,
        "train": len(train_stems),
        "val": len(val_stems),
        "skipped": skipped,
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
    shutdown_after: bool = False


@app.post("/api/training/start")
def training_start(req: TrainRequest):
    # Auto-generate YAML from the selected dataset
    yaml_result = dataset_manager.generate_training_yaml(req.dataset)
    if not yaml_result.get("success"):
        return {"success": False, "error": yaml_result.get("error", "Failed to generate YAML.")}

    trainer.shutdown_after = req.shutdown_after
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


class ClassesUpdateRequest(BaseModel):
    classes: List[str]


@app.post("/api/dataset/{name}/classes")
def dataset_update_classes(name: str, req: ClassesUpdateRequest):
    success = dataset_manager.save_classes(name, req.classes)
    return {"success": success}


@app.get("/api/dataset/{name}/labels/{split}/{filename}")
def dataset_get_labels(name: str, split: str, filename: str):
    labels = dataset_manager.get_labels(name, split, filename)
    return {"labels": labels}


class LabelsUpdateRequest(BaseModel):
    labels: List[List[float]]


@app.post("/api/dataset/{name}/labels/{split}/{filename}")
def dataset_save_labels(name: str, split: str, filename: str, req: LabelsUpdateRequest):
    success = dataset_manager.save_labels(name, split, filename, req.labels)
    return {"success": success}


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
