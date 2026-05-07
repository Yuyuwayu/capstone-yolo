# 🐟 FishWatch — YOLOv8 Fish Appetite Detection

Sistem cerdas berbasis **FastAPI + Vanilla JS** untuk memantau nafsu makan ikan secara otomatis melalui analisis perilaku gerombolan (*schooling behavior*). Proyek ini memanfaatkan **YOLOv8** untuk mendeteksi posisi ikan dan menghitung jarak rata-rata antar individu sebagai indikator status lapar.

> **Logika Utama**: Ikan yang lapar cenderung menyebar (jarak rata-rata lebih jauh), sedangkan ikan yang kenyang cenderung bergerombol (jarak rata-rata lebih dekat).

## ✨ Fitur Utama (All-in-One Dashboard)

Proyek ini telah berevolusi menjadi platform MLOps *end-to-end* dengan antarmuka web modern (**Ocean Depth UI**):

- 📹 **Live Monitoring**: Pantau status ikan (Lapar/Kenyang), jumlah, dan *distance trend* secara *real-time*.
- 🔌 **Multi-Source Video**: Mendukung USB Webcam, Browser Camera (HP/Laptop), DroidCam (IP Camera), hingga *Video File Upload*.
- 🗂️ **Dataset Manager**: Import gambar atau folder dengan fitur *auto-split* (train/val), *bulk delete*, dan **Merge Datasets** (menggabungkan beberapa dataset jadi satu).
- 🚀 **Auto-Training System**: Latih model YOLOv8 langsung dari browser. Sistem otomatis mendeteksi kelas dan men-generate file YAML konfigurasi.
- 📦 **Model Manager**: Upload *weights* model baru (`.pt`) dan aktifkan (Set Active) tanpa perlu *restart* server.
- 📱 **Mobile Friendly**: Layout responsif, bisa diakses dan memonitor dari HP (via `ngrok` atau LAN).

## 🛠️ Prasyarat & Lingkungan Pengembangan

Proyek ini dikelola menggunakan **Poetry** untuk menjamin konsistensi *environment*.

- **Python Version**: `3.13.x` (Sangat disarankan menggunakan `pyenv`)
- **Dependency Manager**: Poetry
- **Hardware Minimum**: CPU modern (Ryzen 5 / i5), 16GB RAM. (GPU NVIDIA disarankan untuk performa *training* dan *monitoring* yang lebih cepat).

## 📥 Instalasi (Replikasi Environment)

Pastikan Anda sudah menginstal [Poetry](https://python-poetry.org/docs/) di sistem Anda.

1. **Clone Repositori**
   ```bash
   git clone https://github.com/Yuyuwayu/capstone-yolo.git
   cd capstone-yolo
   ```

2. **Setup Python & Instal Dependensi**
   Jika menggunakan `pyenv`, set versi Python terlebih dahulu:
   ```bash
   pyenv local 3.13.7
   poetry env use 3.13.7
   ```
   Lalu, instal semua dependensi yang ada di `poetry.lock`:
   ```bash
   poetry install
   ```

## 🎮 Cara Menjalankan FishWatch

Jalankan server FastAPI terintegrasi menggunakan perintah berikut:

```bash
poetry run uvicorn app.server:app --host 0.0.0.0 --port 8000
```

Buka browser Anda dan akses: **`http://localhost:8000`**

### Panduan Singkat Menu Dashboard:
1. **Monitor**: Tab utama untuk melihat live feed kamera, status ikan, dan grafik tren jarak antar ikan.
2. **Dataset**: Tempat manajemen data latih. Anda bisa *import* gambar/folder, mengatur rasio *split*, atau menggabungkan dataset (*merge*).
3. **Training**: Pilih dataset, atur epochs/batch size, dan jalankan proses *training* YOLOv8 tanpa perlu menyentuh terminal.
4. **Models**: Daftar model hasil *training*. Anda bisa *upload* model dari luar atau mengaktifkan model terbaik (`best.pt`) untuk digunakan di tab Monitor.

## 📁 Struktur Direktori Utama

| Folder / File | Deskripsi Fungsi |
| :--- | :--- |
| **`app/`** | Kode utama backend FastAPI (`server.py`, `dataset_manager.py`, `video_stream.py`, dll). |
| **`app/static/`** | Frontend aset (HTML, Vanilla JS, CSS dengan tema *Ocean Depth*). |
| **`dataset/`** | Direktori penyimpanan gambar dan label untuk *training* YOLO. |
| **`models/`** | Direktori penyimpanan model / *weights* YOLO (`.pt`). |
| **`runs/`** | Folder *output* bawaan Ultralytics yang berisi log dan *weights* hasil *training*. |
| **`pyproject.toml`** | "Jantung" proyek; mendefinisikan pustaka yang digunakan oleh Poetry. |
| **`poetry.lock`** | Gembok versi; menyimpan kode *hash* presisi dari setiap dependensi. |

---
*Dibuat untuk mempermudah pemantauan budidaya ikan menggunakan AI.*