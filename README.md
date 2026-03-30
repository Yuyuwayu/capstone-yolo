# Deteksi Nafsu Makan Ikan Berdasarkan Perilaku Menggunakan YOLOv8

Proyek ini adalah sistem cerdas untuk memantau nafsu makan ikan secara otomatis dengan menganalisis perilaku gerombolan (*schooling behavior*) menggunakan deteksi objek YOLOv8. Sistem menyimpulkan bahwa ikan yang lapar cenderung menyebar (jarak antar ikan jauh), sedangkan ikan yang kenyang akan bergerombol (jarak antar ikan dekat).

Analisis ini dilakukan secara *real-time* dari sumber video, dan hasilnya disajikan melalui API serta live video stream.

##  workflow Proyek

Sistem ini bekerja melalui beberapa tahapan utama, mulai dari persiapan data hingga penyajian hasil secara *real-time*.

1.  **Ekstraksi Frame**: Gambar-gambar individual diekstraksi dari video sumber menggunakan `extract.py`.
2.  **Pemisahan Dataset**: Gambar dibagi secara acak menjadi set data latih (80%) dan validasi (20%) menggunakan `auto.py`.
3.  **Anotasi Gambar**: Lokasi ikan ditandai pada setiap gambar. Proyek ini menyediakan dua metode:
    * **Otomatis**: Menggunakan `labeling.py` yang mendeteksi kontur objek.
    * **Semi-otomatis**: Menggunakan `labeling_manual.py`, sebuah alat bantu grafis untuk menambah, menghapus, atau memperbaiki *bounding box* secara manual.
4.  **Pelatihan Model**: Model YOLOv8 dilatih dengan dataset yang telah dianotasi untuk mengenali objek 'ikan'.
5.  **Deteksi & Analisis**: Model yang telah dilatih digunakan untuk mendeteksi ikan dari sumber video. Jarak rata-rata antara semua ikan yang terdeteksi dihitung. Status **"LAPAR"** diberikan jika jarak rata-rata di bawah ambang batas, dan **"Tidak Lapar"** jika sebaliknya.
6.  **Penyajian Hasil**: Hasil deteksi dapat diakses melalui:
    * **REST API**: Endpoint yang menyediakan status deteksi, jarak rata-rata, dan gambar dalam format base64.
    * **Video Stream**: Endpoint untuk streaming video hasil deteksi secara langsung di browser.
    * **Jendela Pratinjau**: Menampilkan video dengan anotasi secara langsung di jendela desktop.

## ✨ Fitur Utama

* **Deteksi Objek Cepat**: Menggunakan arsitektur YOLOv8 yang modern untuk deteksi ikan secara *real-time*.
* **Analisis Perilaku**: Mengukur jarak rata-rata antar ikan sebagai indikator utama nafsu makan.
* **Anotasi Semi-Otomatis**: Alat bantu untuk mempercepat proses pelabelan data dengan antarmuka manual yang interaktif.
* **API & Streaming**: Dibangun dengan FastAPI untuk menyediakan endpoint API dan streaming video, memudahkan integrasi dengan sistem lain.
* **Sumber Fleksibel**: Dapat menggunakan input dari webcam, file video, atau stream video jaringan.

## 📁 Deskripsi File

| Nama File | Deskripsi |
| :--- | :--- |
| **`main.py`** | Aplikasi utama berbasis FastAPI yang menjalankan server untuk API (`/deteksi`) dan live stream video (`/video_feed`). |
| **`extract.py`** | Mengekstrak frame dari file video dengan interval waktu yang bisa diatur (misalnya, satu frame per detik). |
| **`auto.py`** | Memisahkan dataset gambar secara acak ke dalam folder `train` dan `val` dengan rasio 80:20. |
| **`labeling.py`** | Melakukan anotasi (pelabelan) gambar secara otomatis menggunakan metode Computer Vision seperti thresholding dan contour detection. |
| **`labeling_manual.py`** | Menyediakan antarmuka grafis untuk anotasi manual. Pengguna dapat menggambar, menghapus, dan menavigasi antar gambar. |
| **`final.py`** | Skrip untuk menjalankan inferensi model pada video dan menyimpan hasilnya sebagai file video baru (`video_output.mp4`). |
| **`final2.py`** | Skrip inferensi yang lebih canggih, fokus pada analisis jarak antar ikan untuk menentukan status "Lapar" atau "Tidak Lapar". |
| **`testing camera.py`** | Utilitas sederhana untuk memeriksa indeks kamera yang tersedia dan terdeteksi oleh sistem. |

## ⚙️ Prasyarat

* Python 3.8+
* Pustaka Python: `fastapi`, `uvicorn`, `ultralytics`, `opencv-python`, `numpy`, `scipy`.

## 🚀 Instalasi

1.  **Clone Repositori**
    ```bash
    git clone https://github.com/Yuyuwayu/capstone-yolo.git
    cd nama-repositori
    ```

2.  **Buat Virtual Environment (Sangat Disarankan)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal Dependensi**
    ```bash
    pip install fastapi uvicorn "ultralytics[cv2]" numpy scipy
    ```

4.  **Siapkan Model**
    Pastikan Anda memiliki file bobot model YOLOv8 yang telah dilatih (misalnya `best.pt`). Letakkan di dalam direktori yang sesuai seperti `runs/detect/train3/weights/`.

## 🎮 Panduan Penggunaan

### 1. Persiapan Data (Jika Melatih Ulang)

1.  **Kumpulkan Video**: Simpan file video Anda di dalam folder `videos/`.
2.  **Ekstrak Frame**: Jalankan skrip `extract.py`.
    ```bash
    python extract.py
    ```
3.  **Bagi Dataset**: Jalankan `auto.py` untuk membuat set data latih dan validasi.
    ```bash
    python auto.py
    ```
4.  **Anotasi Gambar**: Gunakan `labeling_manual.py` untuk anotasi yang lebih akurat.
    ```bash
    python labeling_manual.py
    ```
    **Kontrol Manual:**
    * `s`: Simpan anotasi & lanjut ke gambar berikutnya.
    * `n`: Lanjut tanpa menyimpan.
    * `p`: Kembali ke gambar sebelumnya.
    * `h`: Beralih antara mode Tambah dan Hapus.
    * `q` atau `Esc`: Keluar.

### 2. Pelatihan Model

Gunakan `ultralytics` untuk melatih model Anda. Pastikan file konfigurasi dataset (`.yaml`) sudah benar.
```bash
yolo task=detect mode=train model=yolov8n.pt data=path/to/data.yaml epochs=100 imgsz=640
```

### 3. Menjalankan Aplikasi

#### a. Menjalankan Server API dan Streaming

Untuk menjalankan server utama, gunakan `uvicorn`.

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Endpoint yang tersedia:
* **API Status**: `http://127.0.0.1:8000/deteksi`
* **Video Stream**: `http://127.0.0.1:8000/video_feed`

#### b. Menjalankan Deteksi pada File Video

Untuk memproses file video lokal dan menampilkan hasilnya, gunakan `final2.py`.
```bash
python final2.py
```

Tekan `q` untuk menghentikan pemutaran video.
