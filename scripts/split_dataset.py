import os
import shutil
import random

# Folder gambar awal
image_dir = 'dataset/images/all'

# Folder tujuan
train_img_dir = 'dataset/images/train'
val_img_dir = 'dataset/images/val'
train_lbl_dir = 'dataset/labels/train'
val_lbl_dir = 'dataset/labels/val'

# Buat semua folder tujuan jika belum ada
for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Ambil semua file .jpg
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
random.shuffle(image_files)

# Pisahkan 80% train, 20% val
val_count = int(0.2 * len(image_files))
val_files = image_files[:val_count]
train_files = image_files[val_count:]

def copy_images(files, dest_dir):
    for f in files:
        src = os.path.join(image_dir, f)
        dst = os.path.join(dest_dir, f)
        shutil.copyfile(src, dst)

# Salin gambar ke folder masing-masing
copy_images(train_files, train_img_dir)
copy_images(val_files, val_img_dir)

print(f"[INFO] Gambar dipisah: {len(train_files)} train, {len(val_files)} val")
