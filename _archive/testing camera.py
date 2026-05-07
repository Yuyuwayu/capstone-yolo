import cv2

def cek_kamera(max_cams=10):
    available_cams = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Kamera ditemukan di index: {i}")
                available_cams.append(i)
            cap.release()
    if not available_cams:
        print("Tidak ada kamera yang ditemukan.")
    return available_cams

kamera_tersedia = cek_kamera()
print("Index kamera yang bisa digunakan:", kamera_tersedia)
