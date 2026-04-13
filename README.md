# 🐟 Deteksi Nafsu Makan Ikan Berdasarkan Perilaku (YOLOv8)

Sistem cerdas untuk memantau nafsu makan ikan secara otomatis melalui analisis perilaku gerombolan (*schooling behavior*). Proyek ini memanfaatkan **YOLOv8** untuk mendeteksi posisi ikan dan menghitung jarak rata-rata antar individu sebagai indikator status lapar.

> **Logika Utama**: Ikan yang lapar cenderung menyebar (jarak jauh), sedangkan ikan yang kenyang cenderung bergerombol (jarak dekat).

## 🔄 Alur Kerja Sistem

1.  **Data Preparation**: Ekstraksi frame (`extract.py`) dan pembagian dataset (`auto.py`).
2.  **Labeling**: Anotasi otomatis (`labeling.py`) atau manual (`labeling_manual.py`).
3.  **Training**: Pelatihan model YOLOv8 untuk mengenali objek 'ikan'.
4.  **Analysis**: Menghitung jarak rata-rata antar ikan secara *real-time*.
5.  **Output**: Hasil disajikan via **FastAPI** (REST API & Video Stream) atau jendela pratinjau desktop.

## ✨ Fitur Utama

* **Deteksi Objek Cepat**: Menggunakan arsitektur YOLOv8 yang modern untuk deteksi ikan secara *real-time*.
* **Analisis Perilaku**: Mengukur jarak rata-rata antar ikan sebagai indikator utama.
* **Anotasi Semi-Otomatis**: Alat bantu pelabelan data manual yang interaktif.
* **API & Streaming**: Dibangun dengan FastAPI untuk kemudahan integrasi.
* **Manajemen Dependensi Modern**: Menggunakan Poetry dan pyenv untuk menjamin konsistensi *environment* lintas OS.

## 🛠️ Prasyarat & Lingkungan Pengembangan

Proyek ini dikelola menggunakan **pyenv** dan **Poetry** untuk menjamin konsistensi *environment* secara absolut.

* **Python Version**: 3.13.7 (Dikelola via `pyenv`)
* **Dependency Manager**: Poetry (Menggunakan `pyproject.toml` dan `poetry.lock`)
* **Hardware Minimum**: AMD Ryzen 5 5500U / 16GB RAM (atau setara).

## 🧰 Instalasi Prasyarat Dasar (Jika Belum Memiliki)

Jika komputer Anda belum memiliki **pyenv** dan **Poetry**, ikuti panduan singkat ini sesuai dengan Sistem Operasi Anda sebelum melanjutkan ke tahap instalasi proyek.

### A.Instalasi pyenv (Manajer Versi Python)

**Untuk Pengguna Windows 11:**
1. Buka PowerShell (sebagai Administrator).
2. Jalankan perintah instalasi berikut:
   ```powershell
   Invoke-WebRequest -UseBasicParsing -Uri "[https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1](https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1)" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
   ```
3. Penting (Hanya Windows): Buka Settings > Apps > Advanced app settings > App execution aliases, lalu matikan (Off) untuk App Installer (python.exe dan python3.exe) agar tidak bentrok dengan Microsoft Store.

**Untuk Pengguna Linux (Ubuntu/Arch) & macOS:**
```bash
# Arch Linux (via AUR)
yay -S pyenv

# Ubuntu / Linux Lainnya (via Curl)
curl [https://pyenv.run](https://pyenv.run) | bash
```
(Jangan lupa tambahkan pyenv ke dalam file .bashrc atau .zshrc Anda sesuai instruksi di terminal).

### B. Instalasi Poetry (Manajer Dependensi)
Sangat disarankan menginstal Poetry secara independen (bukan melalui pip biasa).

**Untuk Pengguna Windows 11:**
Buka PowerShell dan jalankan:
```PowerShell
(Invoke-WebRequest -Uri [https://install.python-poetry.org](https://install.python-poetry.org) -UseBasicParsing).Content | py -
```

(Pastikan Anda menambahkan path C:\Users\<NamaUser>\AppData\Roaming\Python\Scripts ke dalam Environment Variables PATH Windows Anda).

**Untuk Pengguna Linux & macOS:**
Buka terminal dan jalankan:
```bash
curl -sSL [https://install.python-poetry.org](https://install.python-poetry.org) | python3 -
```
## 📥 Instalasi dari Nol (Replikasi Environment)

Proyek ini menggunakan **Poetry** untuk memastikan seluruh dependensi (*library*) beserta versi persisnya terkunci dengan rapi. Ini menghindari masalah "di laptop saya jalan, kok di sini error?" saat berpindah *device* atau berkolaborasi.

Ikuti langkah-langkah di bawah ini untuk mereplikasi *environment* proyek secara identik:

### 1. Clone Repositori
Tarik kode dari GitHub dan masuk ke dalam folder proyek.
```bash
git clone [https://github.com/Yuyuwayu/capstone-yolo.git](https://github.com/Yuyuwayu/capstone-yolo.git)
cd capstone-yolo 
```

### Setup Versi Python (via pyenv)
Karena library Machine Learning sangat sensitif terhadap versi Python, proyek ini dikunci menggunakan Python 3.13.7.
Pastikan utilitas pyenv (atau pyenv-win untuk Windows) sudah terpasang di sistem Anda, lalu jalankan:
```bash
# Instal versi Python yang dibutuhkan
pyenv install 3.13.7

# Kunci versi tersebut HANYA untuk folder proyek ini
pyenv local 3.13.7
```

### 3. Inisialisasi dan Instalasi Dependensi (via Poetry)
Sekarang, instruksikan Poetry untuk menggunakan versi Python yang telah disiapkan di atas, lalu biarkan Poetry mengunduh dan menyusun virtual environment berdasarkan file poetry.lock.
```bash
# Beritahu Poetry untuk memakai Python 3.13.7
poetry env use 3.13.7

# Mulai proses instalasi (otomatis membaca poetry.lock)
poetry install
```
> Penting: Konfigurasi pyproject.toml pada proyek ini telah diatur dengan batas python = ">=3.13, <3.14" untuk mengatasi isu inkompatibilitas instalasi pada library torchvision.

### 4. Siapkan Model (Weights)
Pastikan Anda memiliki file bobot model YOLOv8 hasil pelatihan (misalnya best.pt). Secara default, letakkan file tersebut di dalam direktori yang diminta oleh script (contoh: runs/detect/train3/weights/best.pt).

## 🎮 Panduan Penggunaan

Karena proyek ini dibungkus secara aman menggunakan Poetry, **selalu tambahkan awalan `poetry run`** sebelum mengeksekusi skrip Python apa pun dari terminal. 
*(Alternatif: Anda bisa mengetik `poetry shell` terlebih dahulu untuk masuk ke dalam environment, lalu menjalankan skrip secara normal tanpa awalan).*

### 1. Persiapan Data (Jika Ingin Melatih Ulang)

1.  **Siapkan Video**: Masukkan file video ikan Anda ke dalam folder `videos/`.
2.  **Ekstrak Frame Video**: 
    ```bash
    poetry run python extract.py
    ```
3.  **Pisahkan Dataset (Train/Val)**: 
    ```bash
    poetry run python auto.py
    ```
4.  **Anotasi Bounding Box**: 
    Gunakan alat bantu interaktif kami untuk memperbaiki kotak pelabelan.
    ```bash
    poetry run python labeling_manual.py
    ```
    *(Kontrol Keyboard: `s` simpan & lanjut, `n` lewati, `p` kembali, `h` ganti mode Tambah/Hapus, `q` atau `Esc` untuk keluar).*

### 2. Pelatihan Model (Training)

Gunakan perintah bawaan Ultralytics untuk memulai proses *training*. Pastikan file `data.yaml` sudah dikonfigurasi dengan *path* yang benar menuju folder dataset Anda.
```bash
poetry run yolo task=detect mode=train model=yolov8n.pt data=path/to/data.yaml epochs=100 imgsz=640
```
### 3. Eksekusi Program Utama

Sistem ini menyediakan dua antarmuka untuk melihat hasil deteksi nafsu makan ikan:

**A. Melalui Server API & Live Stream (FastAPI)**
Cocok untuk diintegrasikan dengan web atau aplikasi *mobile*.
```bash
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
* Cek Status (JSON): Buka http://127.0.0.1:8000/deteksi
* Live Video Stream: Buka http://127.0.0.1:8000/video_feed

**B. Melalui Pratinjau Desktop (Real-time Window)**
Cocok untuk pengujian cepat dan melihat analisis jarak antar ikan (schooling behavior) langsung di layar monitor.
```bash
poetry run python final2.py
```
> (Tekan tombol q pada keyboard kapan saja untuk menghentikan pemutaran video).
## 📁 Deskripsi File Utama

Berikut adalah kompas untuk navigasi struktur kode dalam proyek ini:

| Nama File / Folder | Deskripsi Fungsi |
| :--- | :--- |
| **`main.py`** | Aplikasi web FastAPI (menyediakan API endpoint dan *video feed*). |
| **`final2.py`** | Mesin deteksi utama; menganalisis *bounding box* dan menghitung jarak Euclidean antar ikan untuk penentuan status lapar. |
| **`final.py`** | Skrip alternatif untuk menjalankan inferensi model pada video dan menyimpan hasilnya sebagai file `.mp4` baru. |
| **`extract.py`** | Utilitas pemotong video menjadi ribuan gambar frame (Dataset *builder*). |
| **`auto.py`** | *Script* pembagi rasio data latih (80%) dan data validasi (20%) secara acak. |
| **`labeling.py`** | Melakukan anotasi (pelabelan) gambar secara otomatis menggunakan metode Computer Vision (kontur & *thresholding*). |
| **`labeling_manual.py`**| GUI (*Graphical User Interface*) interaktif untuk proses anotasi data secara manual (tambah/hapus *bounding box*). |
| **`testing camera.py`**| Utilitas sederhana untuk memeriksa indeks kamera webcam yang tersedia dan terdeteksi oleh sistem. |
| **`pyproject.toml`** | "Jantung" proyek; mendefinisikan versi Python (3.13.x) dan seluruh pustaka yang digunakan oleh Poetry. |
| **`poetry.lock`** | Gembok versi; menyimpan kode *hash* presisi dari setiap dependensi untuk di-*clone* ke PC atau OS lain tanpa *error*. |