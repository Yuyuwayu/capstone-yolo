<div align="center">
  <img src="docs/monitor.png" alt="FishWatch Dashboard" width="100%">
  <h1>🐟 FishWatch</h1>
  <p><strong>YOLOv8 Fish Appetite Detection & MLOps Dashboard</strong></p>
  <p>
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/YOLOv8-FF1493?style=for-the-badge&logo=ultralytics&logoColor=white" alt="YOLOv8">
    <img src="https://img.shields.io/badge/Vanilla_JS-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="Vanilla JS">
    <img src="https://img.shields.io/badge/Poetry-60A5FA?style=for-the-badge&logo=poetry&logoColor=white" alt="Poetry">
  </p>
</div>

---

## 📖 Overview

**FishWatch** adalah sistem cerdas *end-to-end* untuk memantau nafsu makan ikan secara otomatis melalui analisis perilaku gerombolan (*schooling behavior*). 

Proyek ini telah berevolusi dari sekumpulan skrip Python menjadi **Platform MLOps berbasis Web** yang modern, responsif, dan *user-friendly*, menggunakan tema visual **Ocean Depth**.

> **🧠 Logika Utama:**  
> Ikan yang **Lapar** cenderung menyebar (jarak Euclidean rata-rata lebih jauh).  
> Ikan yang **Kenyang** cenderung bergerombol mengelilingi sisa pakan (jarak rata-rata lebih dekat).

---

## ✨ Fitur Utama (All-in-One Dashboard)

### 📹 1. Live Monitoring (Computer Vision)
Pantau status ikan secara *real-time*! Sistem akan mengekstrak metrik jumlah ikan dan jarak antar ikan untuk memprediksi apakah mereka sedang lapar atau kenyang.
* Mendukung sumber kamera yang beragam: **USB Webcam, Browser Camera (HP/Laptop), DroidCam (IP Camera), dan Upload Video File**.
* Grafik tren jarak (*Distance Trend*) interaktif yang di-update secara *live*.

### 🗂️ 2. Dataset Manager
Tidak perlu lagi memindahkan file secara manual atau membuat struktur *train/val*.
* **Smart Import**: Upload gambar atau *1 folder penuh* langsung dari browser.
* **Auto-Split**: Mengacak dan membagi dataset secara otomatis ke folder `train` dan `val` menggunakan *slider ratio* (misal: 80/20).
* **Merge Datasets**: Gabungkan beberapa dataset berbeda (contoh: `custom` dan `roboflow`) menjadi satu dataset baru yang siap di-training.

### 🚀 3. Auto-Training System
Jalankan proses pelatihan (Training) model YOLOv8 tanpa perlu menyentuh CLI/Terminal.
* Sistem otomatis mendeteksi kelas dari label (misal: `ikan`, `pakan`) dan men-generate file konfigurasi YAML.
* Atur parameter krusial seperti **Epochs, Batch Size, Image Size**, dan **Device** (CPU/GPU) langsung dari antarmuka Web.

### 📦 4. Model Manager
Sistem manajemen *weights* (file `.pt`).
* Lihat daftar model yang tersedia beserta ukurannya.
* Set Active: Pilih model terbaik (`best.pt`) untuk langsung digunakan di *Live Monitoring* tanpa perlu merestart server.

---

## 🚀 Instalasi & Setup (Replikasi Environment)

Proyek ini menggunakan **Poetry** untuk menjamin konsistensi *environment*, sehingga bebas dari isu *"It works on my machine"*.

### Prasyarat
- **Python Version**: `3.13.x` (Sangat disarankan menggunakan `pyenv`)
- **Dependency Manager**: [Poetry](https://python-poetry.org/docs/)
- **Hardware Minimum**: CPU modern (Ryzen 5 / i5), 16GB RAM. (GPU NVIDIA dengan CUDA sangat disarankan untuk proses *training*).

### Langkah-langkah

1. **Clone Repositori**
   ```bash
   git clone https://github.com/Yuyuwayu/capstone-yolo.git
   cd capstone-yolo
   ```

2. **Setup Python & Instal Dependensi**  
   Jika Anda menggunakan `pyenv`, jalankan:
   ```bash
   pyenv local 3.13.7
   poetry env use 3.13.7
   ```
   Lalu, instal semua dependensi yang terkunci di `poetry.lock`:
   ```bash
   poetry install
   ```

---

## 🎮 Cara Menjalankan Aplikasi

Jalankan server **FastAPI** terintegrasi menggunakan Uvicorn:

```bash
poetry run uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Buka browser Anda dan akses: **`http://localhost:8000`**

*(Pro Tip: Jika Anda ingin mengakses dashboard dari HP, gunakan aplikasi seperti `ngrok` untuk mengekspos port 8000 ke publik).*

---

## 📸 Antarmuka Dashboard

### Dataset Management & Auto-Split
<img src="docs/dataset.png" alt="Dataset Manager" width="80%">

### YOLOv8 Auto-Training
<img src="docs/training.png" alt="Training Interface" width="80%">

---

## 📁 Struktur Folder

| Folder / File | Deskripsi Fungsi |
| :--- | :--- |
| **`app/`** | Kode utama backend FastAPI (`server.py`, `dataset_manager.py`, `trainer.py`, dll). |
| **`app/static/`** | Frontend aset (HTML, Vanilla JS, CSS dengan tema *Ocean Depth*). |
| **`dataset/`** | Tempat penyimpanan gambar & label yang dikelola oleh *Dataset Manager*. |
| **`docs/`** | Aset gambar untuk dokumentasi README. |
| **`models/`** | Direktori *weights* model YOLO (`.pt`). |
| **`runs/`** | Folder *output* bawaan Ultralytics (berisi hasil evaluasi, grafik loss, dan *weights* dari training). |
| **`scripts/`** | Kumpulan *standalone script* lama (misal: ekstraksi frame, anotasi manual). |
| **`pyproject.toml`** | "Jantung" proyek; mendefinisikan struktur *library* yang digunakan. |

---
<div align="center">
  <em>Dibuat untuk mempermudah digitalisasi dan pemantauan budidaya perikanan cerdas menggunakan AI.</em>
</div>