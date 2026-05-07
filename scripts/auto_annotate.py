import cv2
import os

def auto_annotate(image_dir, label_dir, class_id=0):
    os.makedirs(label_dir, exist_ok=True)
    for filename in os.listdir(image_dir):
        if not filename.endswith(".jpg"):
            continue

        img_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = img.shape[:2]
        with open(label_path, 'w') as f:
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)

                # Skipping very small blobs (noise)
                if bw * bh < 500:
                    continue

                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                norm_w = bw / w
                norm_h = bh / h

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        print(f"Annotated: {filename}")

auto_annotate("dataset/images/train", "dataset/labels/train")
auto_annotate("dataset/images/val", "dataset/labels/val")
