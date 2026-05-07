"""
FishWatch — Model Manager

Lists, activates, deletes trained YOLO model runs in runs/detect/.
"""

import os
import csv
import shutil
from datetime import datetime

from . import config


class ModelManager:
    def __init__(self):
        self.active_model = config.DEFAULT_MODEL_RUN

    # ── List ──────────────────────────────────────────

    def list_models(self):
        models = []
        if not os.path.isdir(config.RUNS_DIR):
            return models

        for name in sorted(os.listdir(config.RUNS_DIR)):
            run_path = os.path.join(config.RUNS_DIR, name)
            if not os.path.isdir(run_path):
                continue

            weights = os.path.join(run_path, "weights", "best.pt")
            if not os.path.isfile(weights):
                continue

            stat = os.stat(weights)
            args = self._parse_args(run_path)
            metrics = self._last_metrics(run_path)

            models.append({
                "name": name,
                "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d"),
                "size_mb": round(stat.st_size / 1_048_576, 1),
                "epochs": args.get("epochs", "?"),
                "base_model": args.get("model", "?"),
                "dataset": args.get("data", "?"),
                "img_size": args.get("imgsz", "?"),
                "metrics": metrics,
                "is_active": name == self.active_model,
                "path": weights,
            })
        return models

    # ── Activate / Delete ─────────────────────────────

    def set_active(self, run_name):
        weights = os.path.join(config.RUNS_DIR, run_name, "weights", "best.pt")
        if not os.path.isfile(weights):
            return False
        self.active_model = run_name
        return True

    def get_active_path(self):
        return os.path.join(config.RUNS_DIR, self.active_model, "weights", "best.pt")

    def delete_model(self, run_name):
        if run_name == self.active_model:
            return {"success": False, "error": "Cannot delete the active model. Activate another first."}
        run_path = os.path.join(config.RUNS_DIR, run_name)
        if not os.path.isdir(run_path):
            return {"success": False, "error": "Model not found."}
        try:
            shutil.rmtree(run_path)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── Training Curves ───────────────────────────────

    def get_curves(self, run_name):
        """Return per-epoch metrics for charting."""
        results_path = os.path.join(config.RUNS_DIR, run_name, "results.csv")
        if not os.path.isfile(results_path):
            return None

        try:
            with open(results_path, "r") as f:
                reader = csv.reader(f)
                headers = [h.strip() for h in next(reader)]
                epochs = []
                for row in reader:
                    vals = [v.strip() for v in row]
                    pt = {"epoch": len(epochs) + 1}
                    for i, h in enumerate(headers):
                        if i >= len(vals):
                            continue
                        try:
                            v = float(vals[i])
                        except ValueError:
                            continue
                        hl = h.lower().strip()
                        if "box_loss" in hl:
                            pt["box_loss"] = round(v, 4)
                        elif "mAP50" in h and "mAP50-95" not in h:
                            pt["mAP50"] = round(v, 4)
                        elif "mAP50-95" in h:
                            pt["mAP50_95"] = round(v, 4)
                    epochs.append(pt)
                return epochs
        except Exception:
            return None

    # ── Internal Parsers ──────────────────────────────

    @staticmethod
    def _parse_args(run_path):
        path = os.path.join(run_path, "args.yaml")
        if not os.path.isfile(path):
            return {}
        out = {}
        try:
            with open(path, "r") as f:
                for line in f:
                    if ":" not in line:
                        continue
                    k, _, v = line.partition(":")
                    k, v = k.strip(), v.strip()
                    if k in ("epochs", "imgsz", "batch"):
                        try:
                            out[k] = int(v)
                        except ValueError:
                            out[k] = v
                    elif k in ("model", "data"):
                        out[k] = v
        except Exception:
            pass
        return out

    @staticmethod
    def _last_metrics(run_path):
        path = os.path.join(run_path, "results.csv")
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r") as f:
                reader = csv.reader(f)
                headers = [h.strip() for h in next(reader)]
                last = None
                for row in reader:
                    last = row
                if last is None:
                    return {}
                vals = [v.strip() for v in last]
                m = {}
                for i, h in enumerate(headers):
                    if i >= len(vals):
                        continue
                    try:
                        v = float(vals[i])
                    except ValueError:
                        continue
                    if "mAP50" in h and "mAP50-95" not in h:
                        m["mAP50"] = round(v, 4)
                    elif "precision" in h.lower():
                        m["precision"] = round(v, 4)
                    elif "recall" in h.lower():
                        m["recall"] = round(v, 4)
                return m
        except Exception:
            return {}
