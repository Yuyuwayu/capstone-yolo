import cv2
import os

# ==== Konfigurasi ====
image_dir = 'dataset/images/val'
label_dir = 'dataset/labels/val'

# ==== Konfigurasi Kelas ====
CLASSES = {0: "ikan", 1: "pakan"}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)} # Hijau: ikan, Merah: pakan

def preview_annotations():
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    if not img_files:
        print(f"[!] Tidak ada gambar di folder: {image_dir}")
        return

    index = 0
    while 0 <= index < len(img_files):
        fname = img_files[index]
        img_path = os.path.join(image_dir, fname)
        label_path = os.path.join(label_dir, fname.replace('.jpg', '.txt'))
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Gagal membaca gambar: {fname}")
            index += 1
            continue

        h, w = img.shape[:2]
        
        # Mengecek apakah file label (anotasi) ada
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])
                    
                    # Denormalisasi koordinat YOLO kembali ke Piksel
                    x1 = int((x_center - box_width / 2) * w)
                    y1 = int((y_center - box_height / 2) * h)
                    x2 = int((x_center + box_width / 2) * w)
                    y2 = int((y_center + box_height / 2) * h)
                    
                    # Menghindari koordinat keluar batas gambar
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Menggambar Bounding Box
                    color = COLORS.get(class_id, (255, 255, 255)) # Default putih jika class_id aneh
                    label = CLASSES.get(class_id, f"Class {class_id}")
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, max(y1 - 5, 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # Jika belum dianotasi, tampilkan teks peringatan
            cv2.putText(img, "[BELUM ADA ANOTASI]", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # Menampilkan UI teks bantuan
        ui_text = f"Preview: {fname} [{index+1}/{len(img_files)}] | 'n': Next | 'p': Prev | 'q': Quit"
        cv2.putText(img, ui_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Tampilkan Gambar
        window_name = "YOLO Dataset Preview"
        cv2.imshow(window_name, img)
        
        # Kontrol Navigasi
        key = cv2.waitKey(0) & 0xFF
        if key == ord('n'):   # Next
            index = min(len(img_files) - 1, index + 1)
        elif key == ord('p'): # Previous
            index = max(0, index - 1)
        elif key == ord('q') or key == 27: # Quit (Tombol q atau Esc)
            print("[*] Menutup preview...")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("[*] Memulai Preview Dataset...")
    preview_annotations()