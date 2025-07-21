import cv2
import os


def extract_frames_by_seconds(video_dir, output_dir, seconds_interval=1):
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    total_saved = 0
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[ERROR] Failed to open video: {video_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * seconds_interval)
        frame_count = 0
        saved_count = 0
        video_name = os.path.splitext(video_file)[0]
        print(f"[INFO] Processing {video_file} at {fps:.2f} FPS...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                filename = os.path.join(output_dir, f"{video_name}_frame_{saved_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                saved_count += 1
                total_saved += 1

            frame_count += 1

        cap.release()
        print(f"[INFO] {saved_count} frames saved from {video_file}")

    print(f"[INFO] Total {total_saved} frames saved from all videos.")


# Contoh: Ambil 1 frame setiap 1 detik
extract_frames_by_seconds("videos", "dataset/images/all", seconds_interval=1)
