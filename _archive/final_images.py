from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")

img = cv2.imread("train/images/0a8d9a25-dd66-4e24-9acf-f1dc17f31942_jpg.rf.11775faa9e2dfb687a4bfce3fde58c9e.jpg")
img = cv2.resize(img, (640, 480))

results = model(img)

result = results[0]

result.show()