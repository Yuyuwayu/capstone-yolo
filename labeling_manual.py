import cv2
import os

# ==== Konfigurasi ====
image_dir = 'dataset/images/val'
label_dir = 'dataset/labels/val'
class_id = 0
os.makedirs(label_dir, exist_ok=True)

# ==== Global ====
drawing = False
deleting = False
ix, iy = -1, -1
boxes = []
current_img = None
img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
index = 0

def auto_detect(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #thresh = cv2.adaptiveThreshold(blurred, 255,
    #                              cv2.ADAPTIVE_THRESH_MEAN_C,
    #                               cv2.THRESH_BINARY_INV, 11, 3)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    #contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    edges = cv2.Canny(enhanced, 50, 150)

    # Perbesar bentuk ikan
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

        #if area < 500 or aspect > 5 or aspect < 0.2:
        #    continue
        auto_boxes.append(((x, y), (x + bw, y + bh)))
    print(f"[AUTO] Detected {len(auto_boxes)} boxes.")
    return auto_boxes

def mouse_handler(event, x, y, flags, param):
    global drawing, deleting, ix, iy, boxes

    if deleting:
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, ((x1, y1), (x2, y2)) in enumerate(boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    print(f"[✖] Deleted box #{i}")
                    del boxes[i]
                    break
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp = current_img.copy()
        for idx, b in enumerate(boxes):
            cv2.rectangle(temp, b[0], b[1], (0, 255, 0), 2)
            cv2.putText(temp, str(idx), b[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.rectangle(temp, (ix, iy), (x, y), (255, 0, 0), 2)
        cv2.imshow(window_name, temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        boxes.append(((ix, iy), (x, y)))
        print(f"[+] Added box: {((ix, iy), (x, y))}")

def normalize(box, w, h):
    (x1, y1), (x2, y2) = box
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    x_center = ((x1 + x2) / 2) / w
    y_center = ((y1 + y2) / 2) / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def semi_auto_annotate():
    global current_img, boxes, deleting, index, window_name

    while 0 <= index < len(img_files):
        fname = img_files[index]
        img_path = os.path.join(image_dir, fname)
        current_img = cv2.imread(img_path)
        if current_img is None:
            print(f"[!] Failed to read {fname}")
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
            for idx, b in enumerate(boxes):
                cv2.rectangle(temp, b[0], b[1], (0, 255, 0), 2)
                size = f"{b[1][0]-b[0][0]}x{b[1][1]-b[0][1]}"
                label = f"{idx} {size}"
                cv2.putText(temp, label, b[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            mode_text = "[DELETE MODE]" if deleting else "[ADD MODE]"
            cv2.putText(temp, mode_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
            cv2.imshow(window_name, temp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                label_path = os.path.join(label_dir, fname.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    for b in boxes:
                        f.write(normalize(b, w, h) + '\n')
                print(f"[✔] Saved: {label_path}")
                index += 1
                break
            elif key == ord('h'):
                deleting = not deleting
                print(f"[🛠] Delete mode: {deleting}")
            elif key == ord('q') or key == 27:
                print("[×] Exiting...")
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
