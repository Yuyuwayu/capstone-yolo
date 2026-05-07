import cv2
import os

# ==== Konfigurasi ====
image_dir = 'dataset/images/train'
label_dir = 'dataset/labels/train'
os.makedirs(label_dir, exist_ok=True)

# ==== Konfigurasi Kelas ====
CLASSES = {0: "ikan", 1: "pakan"}
COLORS = {0: (0, 255, 0), 1: (0, 0, 255)} # Hijau untuk ikan, Merah untuk pakan
current_class_id = 0 # Default mulai dari kelas 0 (ikan)

# ==== Global ====
drawing = False
deleting = False
ix, iy = -1, -1
boxes = [] # Menyimpan tuple: (class_id, ((x1, y1), (x2, y2)))
current_img = None
img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
index = 0

def print_instructions():
    """Menampilkan tata cara penggunaan di console/terminal"""
    print("\n" + "="*55)
    print(" 🐟 TATA CARA PENGGUNAAN SEMI-AUTO ANNOTATOR 🐟")
    print("="*55)
    print("🖱️  KONTROL MOUSE:")
    print("  - Klik Kiri & Tarik : Menggambar kotak (Mode ADD)")
    print("  - Klik Kiri (tepat di kotak) : Menghapus kotak (Mode DELETE)")
    print("\n⌨️  KONTROL KEYBOARD:")
    print("  [1] : Ganti label aktif menjadi IKAN (Kotak Hijau)")
    print("  [2] : Ganti label aktif menjadi PAKAN (Kotak Merah)")
    print("  [h] : Ganti mode (ADD <--> DELETE)")
    print("  [s] : SIMPAN (Save) anotasi & lanjut ke gambar berikutnya")
    print("  [n] : Lanjut ke gambar berikutnya (Tanpa simpan)")
    print("  [p] : Kembali ke gambar sebelumnya")
    print("  [q] / [Esc] : Keluar dari program")
    print("="*55 + "\n")

def auto_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img.shape[:2]
    auto_boxes = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        aspect = bw / float(bh) if bh != 0 else 0
        if area < 1000 or area > 30000 or aspect < 0.3 or aspect > 4:
            continue

        auto_boxes.append((0, ((x, y), (x + bw, y + bh))))
    return auto_boxes

def mouse_handler(event, x, y, flags, param):
    global drawing, deleting, ix, iy, boxes, current_class_id

    if deleting:
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (cid, ((x1, y1), (x2, y2))) in enumerate(boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    print(f"[✖] Berhasil menghapus kotak #{i} ({CLASSES[cid]})")
                    del boxes[i]
                    break
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = current_img.copy()
        # Gambar kotak yang sudah ada
        for idx, (cid, b) in enumerate(boxes):
            cv2.rectangle(temp, b[0], b[1], COLORS[cid], 2)
            cv2.putText(temp, f"{idx}:{CLASSES[cid]}", b[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[cid], 2)
        
        # Gambar kotak yang sedang ditarik (sesuai warna kelas aktif)
        cv2.rectangle(temp, (ix, iy), (x, y), COLORS[current_class_id], 2)
        cv2.imshow(window_name, temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        boxes.append((current_class_id, ((ix, iy), (x, y))))
        print(f"[+] Menambahkan: {CLASSES[current_class_id].upper()} | Posisi: {((ix, iy), (x, y))}")

def normalize(cid, box, w, h):
    (x1, y1), (x2, y2) = box
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    return f"{cid} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def semi_auto_annotate():
    global current_img, boxes, deleting, index, window_name, current_class_id

    print_instructions() # Panggil instruksi saat program dimulai

    while 0 <= index < len(img_files):
        fname = img_files[index]
        img_path = os.path.join(image_dir, fname)
        current_img = cv2.imread(img_path)
        if current_img is None:
            print(f"[!] Gagal membaca {fname}")
            index += 1
            continue

        h, w = current_img.shape[:2]
        boxes = []
        deleting = False

        window_name = f"Annotator [{index+1}/{len(img_files)}] - {fname}"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_handler)

        while True:
            temp = current_img.copy()
            
            # Draw existing boxes
            for idx, (cid, b) in enumerate(boxes):
                cv2.rectangle(temp, b[0], b[1], COLORS[cid], 2)
                label = f"{CLASSES[cid]}"
                cv2.putText(temp, label, (b[0][0], b[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cid], 2)

            # UI Text Overlay di Gambar
            mode_text = "MODE: [HAPUS KOTAK] (Tekan 'h' utk ke Add Mode)" if deleting else "MODE: [TAMBAH KOTAK] (Tekan 'h' utk ke Delete Mode)"
            mode_color = (0, 165, 255) if deleting else (255, 100, 100) # Orange untuk delete, Biru untuk Add
            class_text = f"KELAS AKTIF: {CLASSES[current_class_id].upper()} (Tekan '1' atau '2' utk ganti)"
            
            # Backgound hitam tipis agar teks selalu terbaca
            cv2.rectangle(temp, (5, 5), (420, 60), (0, 0, 0), -1)
            cv2.putText(temp, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
            cv2.putText(temp, class_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[current_class_id], 1)
            
            cv2.imshow(window_name, temp)

            key = cv2.waitKey(1) & 0xFF
            
            # --- Key Bindings ---
            if key == ord('1'):
                current_class_id = 0
                print("[*] Label aktif diubah ke: IKAN")
            elif key == ord('2'):
                current_class_id = 1
                print("[*] Label aktif diubah ke: PAKAN")
            elif key == ord('s'):
                label_path = os.path.join(label_dir, fname.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    for cid, b in boxes:
                        f.write(normalize(cid, b, w, h) + '\n')
                print(f"[✔] TERSIMPAN: {label_path}")
                index += 1
                break
            elif key == ord('h'):
                deleting = not deleting
                print(f"[🛠] Delete mode aktif: {deleting}")
            elif key == ord('q') or key == 27: # Esc key
                print("[×] Keluar dari program...")
                cv2.destroyAllWindows()
                return
            elif key == ord('p'):
                index = max(0, index - 1)
                break
            elif key == ord('n'):
                index = min(len(img_files) - 1, index + 1)
                break

        cv2.destroyWindow(window_name)

if __name__ == "__main__":
    semi_auto_annotate()