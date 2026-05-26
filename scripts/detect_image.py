from ultralytics import YOLO
import cv2

model = YOLO(r"runs\detect\train18\weights\best.pt")

img = cv2.imread(r"dataset\Robloflow\images\train\IMG_35mm160_00173_jpg.rf.xWVd1SuKLyXafg23lGPy.jpg")
img = cv2.resize(img, (640, 480))

results = model(img)

result = results[0]

result.show()