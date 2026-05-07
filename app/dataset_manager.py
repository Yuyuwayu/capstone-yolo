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
        img_dir = os.path.join(config.DATASET_DIR, dataset, "images", split)
        lbl_dir = os.path.join(config.DATASET_DIR, dataset, "labels", split)
        return img_dir, lbl_dir

    # ── Scan / Stats ──────────────────────────────────

    def scan_datasets(self):
        out = {}
        if not os.path.isdir(config.DATASET_DIR):
            return out
        for ds_name in os.listdir(config.DATASET_DIR):
            ds_base = os.path.join(config.DATASET_DIR, ds_name)
            if not os.path.isdir(ds_base):
                continue
                
            splits = {}
            for split_name in ("train", "val"):
                img_dir, lbl_dir = self._dirs(ds_name, split_name)
                if not os.path.isdir(img_dir):
                    splits[split_name] = {
                        "total": 0,
                        "annotated": 0,
                        "unannotated": 0,
                    }
                    continue
                imgs = self._image_files(img_dir)
                labels = self._label_stems(lbl_dir)
                annotated = sum(1 for f in imgs if os.path.splitext(f)[0] in labels)
                splits[split_name] = {
                    "total": len(imgs),
                    "annotated": annotated,
                    "unannotated": len(imgs) - annotated,
                }
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
        
        dynamic_classes = self.read_classes(dataset)

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
                    label = dynamic_classes.get(cid, f"cls{cid}")
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def get_labels(self, dataset, split, filename):
        """Return list of [class_id, x, y, w, h] for a given image."""
        img_dir, lbl_dir = self._dirs(dataset, split)
        if not lbl_dir:
            return []
        lbl_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")
        labels = []
        if os.path.isfile(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        labels.append([int(parts[0])] + [float(x) for x in parts[1:5]])
        return labels

    def save_labels(self, dataset, split, filename, labels):
        """Save list of [class_id, x, y, w, h] to YOLO txt file."""
        img_dir, lbl_dir = self._dirs(dataset, split)
        if not lbl_dir:
            return False
        lbl_path = os.path.join(lbl_dir, os.path.splitext(filename)[0] + ".txt")
        if not labels:
            if os.path.isfile(lbl_path):
                os.remove(lbl_path)
            return True
        os.makedirs(lbl_dir, exist_ok=True)
        with open(lbl_path, "w") as f:
            for lbl in labels:
                f.write(f"{int(lbl[0])} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
        return True

    # ── Import ────────────────────────────────────────

    def import_folder(self, source_path, dataset_name="custom"):
        """Copy images from an external folder into dataset/images/train."""
        target_dir, _ = self._dirs(dataset_name, "train")

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
        """Shuffle all images (train + val) and re-split."""
        train_img, train_lbl = self._dirs(dataset, "train")
        val_img, val_lbl = self._dirs(dataset, "val")
        
        all_items = []
        for d_img, d_lbl in [(train_img, train_lbl), (val_img, val_lbl)]:
            if os.path.isdir(d_img):
                for f in self._image_files(d_img):
                    lbl = os.path.splitext(f)[0] + ".txt"
                    lbl_path = os.path.join(d_lbl, lbl)
                    all_items.append({
                        "img": os.path.join(d_img, f),
                        "lbl": lbl_path if os.path.isfile(lbl_path) else None,
                        "fname": f,
                        "lname": lbl
                    })
                    
        if not all_items:
            return {"success": False, "error": "No images found to split."}
            
        random.shuffle(all_items)
        val_count = max(1, int(len(all_items) * (1 - train_ratio)))
        val_items = all_items[:val_count]
        train_items = all_items[val_count:]
        
        # Temp dir to avoid overwriting during move
        temp_dir = os.path.join(config.DATASET_DIR, f"{dataset}_temp_split")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Move all to temp
            for idx, item in enumerate(all_items):
                temp_img = os.path.join(temp_dir, f"{idx}_{item['fname']}")
                shutil.move(item["img"], temp_img)
                item["temp_img"] = temp_img
                if item["lbl"]:
                    temp_lbl = os.path.join(temp_dir, f"{idx}_{item['lname']}")
                    shutil.move(item["lbl"], temp_lbl)
                    item["temp_lbl"] = temp_lbl
                    
            # Distribute back
            for items, (d_img, d_lbl) in [(train_items, (train_img, train_lbl)), (val_items, (val_img, val_lbl))]:
                os.makedirs(d_img, exist_ok=True)
                os.makedirs(d_lbl, exist_ok=True)
                for item in items:
                    shutil.move(item["temp_img"], os.path.join(d_img, item["fname"]))
                    if item["lbl"]:
                        shutil.move(item["temp_lbl"], os.path.join(d_lbl, item["lname"]))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        return {"success": True, "train": len(train_items), "val": len(val_items)}

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

        if os.path.isdir(os.path.join(config.DATASET_DIR, target_name)):
            return {"success": False, "error": f"Dataset '{target_name}' already exists."}

        total_copied = 0
        for src_name in source_names:
            for split_name in ("train", "val"):
                src_img, src_lbl = self._dirs(src_name, split_name)
                tgt_img, tgt_lbl = self._dirs(target_name, split_name)

                if os.path.isdir(src_img):
                    os.makedirs(tgt_img, exist_ok=True)
                    os.makedirs(tgt_lbl, exist_ok=True)
                    for f in os.listdir(src_img):
                        if f.lower().endswith((".jpg", ".jpeg", ".png")):
                            src_path = os.path.join(src_img, f)
                            dst_name = f"{src_name}_{f}"
                            shutil.copy2(src_path, os.path.join(tgt_img, dst_name))
                            total_copied += 1

                            lbl_name = os.path.splitext(f)[0] + ".txt"
                            src_lbl_path = os.path.join(src_lbl, lbl_name)
                            if os.path.isfile(src_lbl_path):
                                dst_lbl_name = f"{src_name}_{lbl_name}"
                                shutil.copy2(src_lbl_path, os.path.join(tgt_lbl, dst_lbl_name))

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
        class_ids = set()
        for split_name in ("train", "val"):
            _, lbl_dir = self._dirs(dataset, split_name)
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

    def get_classes_txt_path(self, dataset):
        return os.path.join(config.DATASET_DIR, dataset, "classes.txt")

    def read_classes(self, dataset):
        path = self.get_classes_txt_path(dataset)
        if path and os.path.isfile(path):
            with open(path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]
                return {i: name for i, name in enumerate(lines)}
        return CLASSES

    def save_classes(self, dataset, classes_list):
        path = self.get_classes_txt_path(dataset)
        if not path: return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for c in classes_list:
                f.write(str(c).strip() + "\n")
        return True

    def get_class_names(self, dataset):
        """Return {id: name} mapping for dataset."""
        dynamic_classes = self.read_classes(dataset)
        ids = self.scan_classes(dataset)
        # Ensure all scanned IDs have a name, defaulting to "class_X" if missing
        names = {}
        for cid in ids:
            names[cid] = dynamic_classes.get(cid, f"class_{cid}")
        # Add any classes that are in classes.txt even if not used in labels yet
        for cid, name in dynamic_classes.items():
            if cid not in names:
                names[cid] = name
        return names

    def generate_training_yaml(self, dataset):
        """Auto-generate a data.yaml for YOLO training and return info."""
        class_names = self.get_class_names(dataset)
        if not class_names:
            return {"success": False, "error": "No annotated labels found in dataset."}

        train_img, _ = self._dirs(dataset, "train")
        val_img, _ = self._dirs(dataset, "val")

        if not os.path.isdir(train_img):
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

    # ── Dataset Management ────────────────────────────

    def create_dataset(self, name: str):
        if not name or " " in name or "/" in name or "\\" in name:
            return {"success": False, "error": "Invalid dataset name. Use alphanumeric characters and underscores."}
            
        base = os.path.join(config.DATASET_DIR, name)
        if os.path.exists(base):
            return {"success": False, "error": f"Dataset '{name}' already exists."}
            
        for split in ("train", "val"):
            os.makedirs(os.path.join(base, "images", split), exist_ok=True)
            os.makedirs(os.path.join(base, "labels", split), exist_ok=True)
            
        # Create default classes.txt
        with open(os.path.join(base, "classes.txt"), "w") as f:
            f.write("ikan\npakan\n")
            
        return {"success": True, "dataset": name}
        
    def delete_dataset(self, name: str):
        if not name:
            return {"success": False, "error": "Invalid dataset name."}
            
        base = os.path.join(config.DATASET_DIR, name)
        if not os.path.exists(base):
            return {"success": False, "error": f"Dataset '{name}' not found."}
            
        shutil.rmtree(base, ignore_errors=True)
        return {"success": True}

