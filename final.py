from ultralytics import YOLO
import cv2

# Load model yang sudah dilatih
model = YOLO('runs/detect/train6/weights/best.pt')  # Ganti path sesuai model kamu

# Buka video input
video_path = 1 # Ganti dengan path video kamu
cap = cv2.VideoCapture(video_path)

# Ambil detail video (resolusi, fps, dll)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Siapkan video output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec video
out = cv2.VideoWriter('video_output.mp4', fourcc, fps, (width, height))

# Loop frame demi frame
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Jalankan prediksi YOLO
    results = model(frame, verbose=False)[0]

    # Render hasil ke frame
    annotated_frame = results.plot()  # hasil gambar dengan bbox

    # Tulis frame ke video output
    out.write(annotated_frame)

    # Tampilkan (opsional)
    cv2.imshow('YOLOv8 Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
out.release()
cv2.destroyAllWindows()
