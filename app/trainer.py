"""
FishWatch — YOLO Trainer

Start, monitor, and stop YOLO training from the dashboard.
Training runs in a background thread; logs are streamed via SSE.
"""

import os
import re
import subprocess
import sys
import threading
from collections import deque

from . import config


class YOLOTrainer:
    def __init__(self):
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.log_buffer = deque(maxlen=500)
        self.latest_metrics = {}
        self.new_model_available = False
        self._process = None
        self._thread = None

    # ── Start ─────────────────────────────────────────

    def start_training(self, train_config: dict):
        if self.is_training:
            return {"success": False, "error": "Training already in progress."}

        model = train_config.get("model", config.DEFAULT_BASE_MODEL)
        data = train_config.get("data", "data.yaml")
        epochs = int(train_config.get("epochs", config.DEFAULT_EPOCHS))
        batch = int(train_config.get("batch", config.DEFAULT_BATCH_SIZE))
        imgsz = int(train_config.get("imgsz", config.DEFAULT_IMG_SIZE))
        device = train_config.get("device", "cpu")

        self.total_epochs = epochs
        self.current_epoch = 0
        self.log_buffer.clear()
        self.latest_metrics = {}
        self.new_model_available = False
        self.is_training = True

        self._thread = threading.Thread(
            target=self._run,
            args=(model, data, epochs, batch, imgsz, device),
            daemon=True,
        )
        self._thread.start()

        return {"success": True, "total_epochs": epochs}

    def _run(self, model, data, epochs, batch, imgsz, device):
        yolo_cmd = "yolo.exe" if os.name == "nt" else "yolo"
        yolo_path = os.path.join(os.path.dirname(sys.executable), yolo_cmd)
        
        if not os.path.exists(yolo_path):
            yolo_path = yolo_cmd

        cmd = [
            yolo_path,
            "detect", "train",
            f"model={model}",
            f"data={data}",
            f"epochs={epochs}",
            f"batch={batch}",
            f"imgsz={imgsz}",
            f"device={device}",
        ]

        self.log_buffer.append(f"[CMD] {' '.join(cmd)}")

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=config.BASE_DIR,
                bufsize=1,
            )

            epoch_re = re.compile(r"(\d+)/(\d+)")

            for line in iter(self._process.stdout.readline, ""):
                line = line.rstrip()
                if not line:
                    continue
                self.log_buffer.append(line)

                # Try to parse epoch number
                m = epoch_re.search(line)
                if m:
                    self.current_epoch = int(m.group(1))

            self._process.wait()

            if self._process.returncode == 0:
                self.log_buffer.append("[OK] Training completed successfully.")
                self.new_model_available = True
                self.current_epoch = self.total_epochs
            else:
                self.log_buffer.append(f"[ERROR] Training exited with code {self._process.returncode}")

        except Exception as e:
            self.log_buffer.append(f"[ERROR] {e}")
        finally:
            self.is_training = False
            self._process = None

    # ── Status ────────────────────────────────────────

    def get_status(self):
        return {
            "is_training": self.is_training,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "progress": round(self.current_epoch / self.total_epochs * 100, 1) if self.total_epochs else 0,
            "new_model_available": self.new_model_available,
            "logs": list(self.log_buffer)[-50:],  # last 50 lines
        }

    def get_full_logs(self):
        return list(self.log_buffer)

    # ── Stop ──────────────────────────────────────────

    def stop_training(self):
        if not self.is_training or not self._process:
            return {"success": False, "error": "No training in progress."}
        try:
            self._process.terminate()
            self.log_buffer.append("[INFO] Training stopped by user.")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── List data YAMLs ───────────────────────────────

    @staticmethod
    def list_data_yamls():
        yamls = []
        for f in os.listdir(config.BASE_DIR):
            if f.endswith(".yaml") or f.endswith(".yml"):
                yamls.append(f)
        return yamls
