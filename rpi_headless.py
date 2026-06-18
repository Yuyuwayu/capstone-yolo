"""Minimal FishWatch runtime for Raspberry Pi.

Runs YOLO inference and centroid-distance appetite logic without the web
dashboard, dataset manager, training service, or GUI.
"""

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "detect" / "train18" / "weights" / "best.pt"


def parse_source(value):
    return int(value) if value.isdigit() else value


def average_distance(centroids):
    if len(centroids) < 2:
        return None
    distances = [
        math.dist(centroids[i], centroids[j])
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    ]
    return float(np.mean(distances))


def open_capture(source, width, height, fps):
    if isinstance(source, int):
        capture = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(source)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, fps)
        return capture
    return cv2.VideoCapture(source)


def build_parser():
    parser = argparse.ArgumentParser(description="FishWatch headless Raspberry Pi runtime")
    parser.add_argument("--source", default="0", help="Camera index, video path, or MJPEG URL")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to YOLO best.pt")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--distance-threshold", type=float, default=300.0)
    parser.add_argument("--frame-skip", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--report-interval", type=float, default=1.0)
    parser.add_argument("--loop-video", action="store_true", help="Loop local video files")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames")
    parser.add_argument("--output", help="Save annotated processed frames to MP4")
    parser.add_argument("--show", action="store_true", help="Show an OpenCV window when a desktop is available")
    return parser


def main():
    args = build_parser().parse_args()
    source = parse_source(args.source)
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"Model tidak ditemukan: {model_path}")

    model = YOLO(str(model_path))
    capture = open_capture(source, args.width, args.height, args.fps)
    if not capture.isOpened():
        raise SystemExit(f"Sumber video gagal dibuka: {source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    output_fps = max(1.0, source_fps / max(1, args.frame_skip)) if source_fps > 0 else 5.0
    writer = None
    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    history = deque()
    inference_times = deque(maxlen=30)
    frame_index = 0
    processed_frames = 0
    last_report = 0.0
    last_status = "Menunggu frame valid"

    print(
        f"FishWatch headless | source={source} | imgsz={args.imgsz} | "
        f"conf={args.conf} | skip={args.frame_skip}"
    )

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if args.loop_video and isinstance(source, str) and Path(source).is_file():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            frame_index += 1
            if frame_index % max(1, args.frame_skip) != 0:
                continue

            inference_start = time.perf_counter()
            result = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            inference_times.append(time.perf_counter() - inference_start)
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            confidences = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
            centroids = [
                ((float(box[0]) + float(box[2])) / 2, (float(box[1]) + float(box[3])) / 2)
                for box in boxes
            ]

            now = time.monotonic()
            distance = average_distance(centroids)
            if distance is not None:
                history.append((now, distance))

            cutoff = now - args.window_seconds
            while history and history[0][0] < cutoff:
                history.popleft()

            window_distances = [value for _, value in history]
            window_average = (
                float(np.mean(window_distances))
                if window_distances
                else None
            )
            hungry_percentage = None
            if window_distances:
                hungry_count = sum(
                    value < args.distance_threshold
                    for value in window_distances
                )
                hungry_percentage = 100.0 * hungry_count / len(window_distances)
                last_status = (
                    "Lapar"
                    if hungry_count > len(window_distances) - hungry_count
                    else "Tidak Lapar"
                )

            annotated = frame.copy()
            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (65, 185, 90), 2)
                cv2.putText(
                    annotated,
                    f"fish {float(confidence):.2f}",
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (65, 185, 90),
                    2,
                    cv2.LINE_AA,
                )

            distance_text = "-" if distance is None else f"{distance:.2f}"
            average_text = "-" if window_average is None else f"{window_average:.2f}"
            hungry_text = "-" if hungry_percentage is None else f"{hungry_percentage:.1f}%"
            overlay = (
                f"{last_status} | fish={len(centroids)} | d_avg={distance_text} | "
                f"hungry={hungry_text}"
            )
            cv2.rectangle(annotated, (8, 8), (min(annotated.shape[1] - 8, 720), 45), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                overlay,
                (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if output_path:
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Gagal membuat video output: {output_path}")
                writer.write(annotated)

            if args.show:
                cv2.imshow("FishWatch Raspberry Pi", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed_frames += 1
            if now - last_report >= args.report_interval:
                inference_fps = (
                    len(inference_times) / sum(inference_times)
                    if inference_times and sum(inference_times) > 0
                    else 0.0
                )
                print(
                    f"ikan={len(centroids):2d} | d_avg={distance_text:>7} | "
                    f"window={average_text:>7} | lapar={hungry_text:>6} | "
                    f"status={last_status} | inferensi={inference_fps:.2f} FPS"
                )
                last_report = now

            if args.max_frames and processed_frames >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nDihentikan.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f"Video hasil disimpan: {output_path}")
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
