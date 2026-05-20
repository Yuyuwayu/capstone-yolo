"""
Find and remove files with identical content (same hash) even if different names.
Keeps the first occurrence, removes the rest.
"""
import os
import hashlib
from collections import defaultdict

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "custom")

total_removed_img = 0
total_removed_lbl = 0

for split in ["train", "val"]:
    img_dir = os.path.join(BASE, "images", split)
    lbl_dir = os.path.join(BASE, "labels", split)

    if not os.path.isdir(img_dir):
        print(f"[{split}] images dir not found, skipping")
        continue

    files = sorted(os.listdir(img_dir))
    print(f"\n=== {split} === ({len(files)} images)")

    # Hash all files
    hash_map = defaultdict(list)
    for f in files:
        path = os.path.join(img_dir, f)
        h = hashlib.md5(open(path, "rb").read()).hexdigest()
        hash_map[h].append(f)

    removed_img = 0
    removed_lbl = 0

    for h, group in hash_map.items():
        if len(group) <= 1:
            continue
        keep = group[0]
        for dup in group[1:]:
            img_path = os.path.join(img_dir, dup)
            print(f"  REMOVE (dup of {keep}): {dup}")
            os.remove(img_path)
            removed_img += 1

            # Also remove matching label
            lbl_name = os.path.splitext(dup)[0] + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            if os.path.isfile(lbl_path):
                os.remove(lbl_path)
                removed_lbl += 1

    print(f"  Removed: {removed_img} images, {removed_lbl} labels")
    total_removed_img += removed_img
    total_removed_lbl += removed_lbl

    remaining = len(os.listdir(img_dir))
    lbl_count = len(os.listdir(lbl_dir)) if os.path.isdir(lbl_dir) else 0
    print(f"  Remaining: {remaining} images, {lbl_count} labels")

print(f"\n=== TOTAL REMOVED: {total_removed_img} images, {total_removed_lbl} labels ===")
