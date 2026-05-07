"""
FishWatch — Dataset Manager

Browse, preview, import, split, and delete dataset images.
"""

import os
import random
import shutil

import cv2

from . import config


CLASSES = {0: "ikan", 1: "pakan"}
COLORS = {0: (178, 186, 60), 1: (60, 186, 93)}  # BGR: teal-ish, green


class DatasetManager:

    # ── Resolve Paths ─────────────────────────────────

    def _dirs(self, dataset, split):
        """Return (images_dir, labels_dir) for a given dataset + split."""
        ds = config.DATASETS.get(dataset)
        if not ds:
            return None, None
        sp = ds["splits"].get(split)
        if not sp:
            return None, None
        return sp["images"], sp["labels"]

    # ── Scan / Stats ──────────────────────────────────

    def scan_datasets(self):
        out = {}
        for ds_name, ds_conf in config.DATASETS.items():
            splits = {}
            for split_name, split_paths in ds_conf["splits"].items():
                img_dir = split_paths["images"]
                lbl_dir = split_paths["labels"]
                if not os.path.isdir(img_dir):
                    continue
                imgs = self._image_files(img_dir)
                labels = self._label_stems(lbl_dir)
                annotated = sum(1 for f in imgs if os.path.splitext(f)[0] in labels)
                splits[split_name] = {
                    "total": len(imgs),
                    "annotated": annotated,
                    "unannotated": len(imgs) - annotated,
                }
            if splits:
                out[ds_name] = {"name": ds_name, "splits": splits}
        return out

    # ── List Images ───────────────────────────────────

    def list_images(self, dataset, split, filter_type="all"):
        img_dir, lbl_dir = self._dirs(dataset, split)
        if not img_dir or not os.path.isdir(img_dir):
            return []
        imgs = sorted(self._image_files(img_dir))
        labels = self._label_stems(lbl_dir)

        result = []
        for f in imgs:
            stem = os.path.splitext(f)[0]
            has_label = stem in labels
            if filter_type == "annotated" and not has_label:
                continue
            if filter_type == "unannotated" and has_label:
                continue
            result.append({"filename": f, "has_label": has_label})
        return result

    # ── Serve Image Bytes ─────────────────────────────

    def get_image_path(self, dataset, split, filename):
        img_dir, _ = self._dirs(dataset, split)
        if not img_dir:
            return None
        path = os.path.join(img_dir, filename)
        return path if os.path.isfile(path) else None

    def get_annotated_preview(self, dataset, split, filename):
        """Return JPEG bytes of an image with its YOLO-format bboxes drawn."""
        img_dir, lbl_dir = self._dirs(dataset, split)
        if not img_dir:
            return None

        img_path = os.path.join(img_dir, filename)
        if not os.path.isfile(img_path):
            return None

        img = cv2.imread(img_path)
        if img is None:
            return None

        h, w = img.shape[:2]
        lbl_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")

        if os.path.isfile(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:])
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    color = COLORS.get(cid, (255, 255, 255))
                    label = CLASSES.get(cid, f"cls{cid}")
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    # ── Import ────────────────────────────────────────

    def import_folder(self, source_path, dataset_name="custom"):
        """Copy images from an external folder into dataset/images/all."""
        ds = config.DATASETS.get(dataset_name)
        if not ds:
            return {"success": False, "error": f"Unknown dataset '{dataset_name}'."}

        target_dir = ds["splits"].get("all", ds["splits"].get("train", {})).get("images")
        if not target_dir:
            return {"success": False, "error": "No target split found."}

        if not os.path.isdir(source_path):
            return {"success": False, "error": f"Source path not found: {source_path}"}

        os.makedirs(target_dir, exist_ok=True)
        count = 0
        for f in os.listdir(source_path):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                shutil.copy2(os.path.join(source_path, f), os.path.join(target_dir, f))
                count += 1

        return {"success": True, "imported": count}

    # ── Split ─────────────────────────────────────────

    def split_dataset(self, dataset, train_ratio=0.8):
        """Shuffle images in 'all' and split into train/val."""
        ds = config.DATASETS.get(dataset)
        if not ds:
            return {"success": False, "error": "Unknown dataset."}

        all_split = ds["splits"].get("all")
        train_split = ds["splits"].get("train")
        val_split = ds["splits"].get("val")

        if not all_split or not train_split or not val_split:
            return {"success": False, "error": "Dataset doesn't have all/train/val splits."}

        src_img = all_split["images"]
        src_lbl = all_split["labels"]

        if not os.path.isdir(src_img):
            return {"success": False, "error": "No images in 'all' split."}

        images = self._image_files(src_img)
        random.shuffle(images)

        val_count = max(1, int(len(images) * (1 - train_ratio)))
        val_files = images[:val_count]
        train_files = images[val_count:]

        for split_paths, files in [(train_split, train_files), (val_split, val_files)]:
            os.makedirs(split_paths["images"], exist_ok=True)
            os.makedirs(split_paths["labels"], exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(src_img, f), os.path.join(split_paths["images"], f))
                lbl = os.path.splitext(f)[0] + ".txt"
                lbl_src = os.path.join(src_lbl, lbl) if os.path.isdir(src_lbl) else ""
                if os.path.isfile(lbl_src):
                    shutil.copy2(lbl_src, os.path.join(split_paths["labels"], lbl))

        return {"success": True, "train": len(train_files), "val": len(val_files)}

    # ── Delete ────────────────────────────────────────

    def delete_image(self, dataset, split, filename):
        img_dir, lbl_dir = self._dirs(dataset, split)
        if not img_dir:
            return {"success": False, "error": "Invalid dataset/split."}

        img_path = os.path.join(img_dir, filename)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")

        deleted = []
        if os.path.isfile(img_path):
            os.remove(img_path)
            deleted.append(filename)
        if os.path.isfile(lbl_path):
            os.remove(lbl_path)

        return {"success": True, "deleted": deleted}

    def delete_images_bulk(self, dataset, split, filenames):
        results = []
        for f in filenames:
            results.append(self.delete_image(dataset, split, f))
        deleted = sum(1 for r in results if r["success"])
        return {"success": True, "deleted_count": deleted}
    # ── Merge Datasets ─────────────────────────────────

    def merge_datasets(self, source_names: list[str], target_name: str):
        """Merge images+labels from multiple datasets into a new one."""
        if not source_names or len(source_names) < 2:
            return {"success": False, "error": "Select at least 2 datasets to merge."}

        if target_name in config.DATASETS:
            return {"success": False, "error": f"Dataset '{target_name}' already exists."}

        # Create target dataset structure
        base = os.path.join(config.DATASET_DIR, target_name)
        target_splits = {}
        for split_name in ("train", "val"):
            img_dir = os.path.join(base, split_name, "images")
            lbl_dir = os.path.join(base, split_name, "labels")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            target_splits[split_name] = {"images": img_dir, "labels": lbl_dir}

        # Copy from each source dataset
        total_copied = 0
        for src_name in source_names:
            ds = config.DATASETS.get(src_name)
            if not ds:
                continue

            for split_name, split_paths in ds["splits"].items():
                # Map source split to target (valid → val, test → skip or val)
                target_split = split_name
                if split_name == "valid":
                    target_split = "val"
                if split_name in ("all", "test"):
                    continue  # skip 'all' and 'test'

                if target_split not in target_splits:
                    continue

                src_img = split_paths.get("images", "")
                src_lbl = split_paths.get("labels", "")
                tgt_img = target_splits[target_split]["images"]
                tgt_lbl = target_splits[target_split]["labels"]

                if os.path.isdir(src_img):
                    for f in os.listdir(src_img):
                        if f.lower().endswith((".jpg", ".jpeg", ".png")):
                            src_path = os.path.join(src_img, f)
                            # Prefix filename with source name to avoid collisions
                            dst_name = f"{src_name}_{f}"
                            shutil.copy2(src_path, os.path.join(tgt_img, dst_name))
                            total_copied += 1

                            # Copy matching label
                            lbl_name = os.path.splitext(f)[0] + ".txt"
                            src_lbl_path = os.path.join(src_lbl, lbl_name) if os.path.isdir(src_lbl) else ""
                            if os.path.isfile(src_lbl_path):
                                dst_lbl_name = f"{src_name}_{lbl_name}"
                                shutil.copy2(src_lbl_path, os.path.join(tgt_lbl, dst_lbl_name))

        # Register new dataset in config
        config.DATASETS[target_name] = {
            "base": base,
            "splits": target_splits,
        }

        return {
            "success": True,
            "dataset": target_name,
            "total_images": total_copied,
            "sources": source_names,
        }

    # ── Helpers ────────────────────────────────────────

    @staticmethod
    def _image_files(directory):
        if not os.path.isdir(directory):
            return []
        return [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    @staticmethod
    def _label_stems(directory):
        if not directory or not os.path.isdir(directory):
            return set()
        return {os.path.splitext(f)[0] for f in os.listdir(directory) if f.endswith(".txt")}

    # ── Class Detection & YAML Generation ─────────────

    def scan_classes(self, dataset):
        """Scan all label files across all splits and return unique class IDs."""
        ds = config.DATASETS.get(dataset)
        if not ds:
            return []

        class_ids = set()
        for split_paths in ds["splits"].values():
            lbl_dir = split_paths.get("labels", "")
            if not os.path.isdir(lbl_dir):
                continue
            for fname in os.listdir(lbl_dir):
                if not fname.endswith(".txt"):
                    continue
                with open(os.path.join(lbl_dir, fname), "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_ids.add(int(parts[0]))
                            except ValueError:
                                pass

        return sorted(class_ids)

    def get_class_names(self, dataset):
        """Return {id: name} mapping. Uses known CLASSES, falls back to generic."""
        ids = self.scan_classes(dataset)
        names = {}
        for cid in ids:
            names[cid] = CLASSES.get(cid, f"class_{cid}")
        return names

    def generate_training_yaml(self, dataset):
        """Auto-generate a data.yaml for YOLO training and return info."""
        ds = config.DATASETS.get(dataset)
        if not ds:
            return {"success": False, "error": f"Unknown dataset '{dataset}'."}

        class_names = self.get_class_names(dataset)
        if not class_names:
            return {"success": False, "error": "No annotated labels found in dataset."}

        # Determine train/val paths
        train_img = None
        val_img = None
        for sname in ("train",):
            sp = ds["splits"].get(sname)
            if sp and os.path.isdir(sp["images"]):
                train_img = sp["images"]
        for sname in ("val", "valid"):
            sp = ds["splits"].get(sname)
            if sp and os.path.isdir(sp["images"]):
                val_img = sp["images"]

        if not train_img:
            return {"success": False, "error": "No train split with images found."}

        # Build YAML content
        nc = len(class_names)
        names_list = [class_names.get(i, f"class_{i}") for i in range(max(class_names.keys()) + 1)]

        yaml_lines = [
            f"# Auto-generated by FishWatch for dataset: {dataset}",
            f"path: {config.BASE_DIR}",
            f"train: {os.path.relpath(train_img, config.BASE_DIR)}",
        ]
        if val_img:
            yaml_lines.append(f"val: {os.path.relpath(val_img, config.BASE_DIR)}")
        yaml_lines.append("")
        yaml_lines.append(f"nc: {nc}")
        yaml_lines.append(f"names: {names_list}")

        yaml_content = "\n".join(yaml_lines) + "\n"

        # Write to BASE_DIR / {dataset}_auto.yaml
        yaml_filename = f"{dataset}_auto.yaml"
        yaml_path = os.path.join(config.BASE_DIR, yaml_filename)
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        return {
            "success": True,
            "yaml_file": yaml_filename,
            "yaml_path": yaml_path,
            "nc": nc,
            "names": names_list,
            "train_images": train_img,
            "val_images": val_img,
        }
