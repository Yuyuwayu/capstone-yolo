# Menjalankan FishWatch di Raspberry Pi

## Rekomendasi

Gunakan Raspberry Pi 4 hanya untuk inference/demo ringan. Training model tetap dilakukan di laptop/PC, lalu file `best.pt` dipakai di Raspberry Pi.

## Preset ringan

Preset default aplikasi sudah dibuat ringan:

- Ukuran inferensi YOLO: `320`
- Proses setiap 3 frame: `PROCESS_EVERY_N_FRAMES=3`
- Resolusi kamera: `640x480`
- FPS kamera: `10`
- JPEG quality stream: `65`

## Jalankan di Raspberry Pi

```bash
cd ~/capstone-yolo
sudo apt update
sudo apt install -y python3-venv python3-numpy python3-opencv \
  python3-scipy python3-matplotlib python3-torch python3-torchvision

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-rpi.txt
pip install --no-cache-dir --no-deps ultralytics
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Tes seluruh dependency runtime:

```bash
python -c "import numpy, cv2, scipy, matplotlib, torch, torchvision; print('native packages OK')"
python -c "import polars, thop; from ultralytics import YOLO; print('YOLO runtime OK')"
python -c "import fastapi, uvicorn, multipart; print('web runtime OK')"
```

Buka dari browser:

```text
http://IP_RASPBERRY_PI:8000
```

## Jika masih berat

Turunkan lagi setting lewat environment variable:

```bash
export FISHWATCH_INFERENCE_IMAGE_SIZE=256
export FISHWATCH_PROCESS_EVERY_N_FRAMES=5
export FISHWATCH_CAMERA_WIDTH=480
export FISHWATCH_CAMERA_HEIGHT=360
export FISHWATCH_CAMERA_FPS=6
export FISHWATCH_STREAM_JPEG_QUALITY=55
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000
```

## Catatan

- Jangan training di Raspberry Pi.
- Jangan pakai `requirements.txt` laptop di Raspberry Pi. Pakai `requirements-rpi.txt`.
- Pakai heatsink/fan karena YOLO akan membuat CPU panas.
- Jika kamera Pi tidak muncul sebagai webcam biasa, gunakan source browser camera dari dashboard atau pastikan kamera tersedia sebagai `/dev/video0`.

## Jika muncul `Illegal instruction`

Biasanya penyebabnya adalah wheel Python yang tidak cocok dengan CPU Raspberry Pi. Cek modul mana yang crash:

```bash
source .venv/bin/activate
python -X faulthandler -c "import numpy; print('numpy ok')"
python -X faulthandler -c "import cv2; print('cv2 ok')"
python -X faulthandler -c "import torch; print('torch ok', torch.__version__)"
python -X faulthandler -c "from ultralytics import YOLO; print('ultralytics ok')"
```

Jika crash di `cv2`, gunakan OpenCV dari apt:

```bash
sudo apt update
sudo apt install -y python3-opencv
python3 -m venv --system-site-packages .venv
```

Jika crash di `torch`, hapus torch dari venv dan pasang wheel PyTorch yang cocok untuk arsitektur Raspberry Pi yang digunakan. Pastikan OS Raspberry Pi 64-bit:

```bash
getconf LONG_BIT
uname -m
```

Hasil yang disarankan:

```text
64
aarch64
```
