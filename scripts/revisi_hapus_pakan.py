"""
Revisi Skripsi — Menghapus Indikator Pakan

Script ini mengedit file 'Skripsi (1).docx' secara otomatis,
mengganti semua referensi 'pakan' / 'pellet' sesuai panduan revisi.

Output: 'Skripsi (REVISI).docx' — file asli TIDAK diubah.
"""

import copy
import os
import re
import sys

from docx import Document


# ═══════════════════════════════════════════════════════════════
# DAFTAR PENGGANTIAN TEKS
# Format: (teks_lama, teks_baru)
# Urutan PENTING — penggantian yang lebih spesifik (lebih panjang) harus di atas
# agar tidak tertimpa oleh penggantian yang lebih umum.
# ═══════════════════════════════════════════════════════════════

REPLACEMENTS = [
    # ── Judul Skripsi (di biodata/lampiran) ──
    (
        "Sistem deteksi tingkat nafsu makan ikan mujair berbasis analisis perilaku agregasi dan sisa pakan menggunakan Yolov8",
        "Sistem deteksi tingkat nafsu makan ikan mujair berbasis analisis perilaku agregasi menggunakan YOLOv8n"
    ),

    # ── KEBARUAN PENELITIAN (rombak total) ──
    (
        "Kebaruan utama (novelty) dari penelitian ini terletak pada integrasi dua parameter deteksi\u2014perilaku agregasi biologis dan sisa pakan fisik\u2014menggunakan arsitektur deep learning ringan (YOLOv8n) pada lingkungan eksperimental dengan sumber daya terbatas.",
        "Kebaruan utama (novelty) dari penelitian ini terletak pada penerapan arsitektur deep learning ultra-ringan YOLOv8n (Nano) untuk deteksi perilaku agregasi ikan secara real-time pada perangkat komputasi standar (laptop CPU-only). Berbeda dengan penelitian-penelitian sebelumnya yang umumnya menggunakan model berat dan memerlukan GPU kelas atas, penelitian ini membuktikan bahwa analisis pola distribusi spasial ikan dapat dilakukan secara akurat dengan model nano yang memiliki jumlah parameter jauh lebih kecil, sehingga lebih terjangkau dan dapat direplikasi oleh pembudidaya skala kecil."
    ),
    (
        "Berbeda dengan penelitian terdahulu yang umumnya mengevaluasi nafsu makan hanya berdasarkan satu indikator (misalnya: hanya perilaku renang atau hanya sisa pakan secara terpisah), penelitian ini mengajukan pendekatan fusi informasi visual. Sistem dirancang untuk mengkorelasikan kepadatan agregasi ikan dengan keberadaan sisa pakan secara simultan untuk menghasilkan keputusan tingkat nafsu makan yang lebih akurat dan objektif.",
        "Berbeda dengan penelitian terdahulu yang umumnya menggunakan model deteksi objek yang berat dan memerlukan perangkat keras kelas atas, penelitian ini mengajukan pendekatan deteksi berbasis fitur makro perilaku agregasi yang tidak memerlukan deteksi objek mikro (butiran pakan), sehingga memungkinkan penggunaan model dengan arsitektur paling ringan tanpa mengorbankan fungsionalitas sistem."
    ),

    # ── LATAR BELAKANG ──
    (
        "Integrasi antara deteksi perilaku agregasi dan pemantauan intensitas makan ini dapat memberikan gambaran komprehensif mengenai status kekenyangan ikan, memungkinkan penghentian pemberian pakan yang tepat waktu untuk meminimalkan limbah.",
        "Deteksi perilaku agregasi ini dapat memberikan gambaran komprehensif mengenai status kekenyangan ikan, memungkinkan penghentian pemberian pakan yang tepat waktu untuk meminimalkan limbah."
    ),

    # ── BATASAN MASALAH ──
    (
        "YOLOv8n (nano) untuk mendeteksi dua kelas objek utama: ikan dan butiran pakan terapung. Tidak dilakukan perbandingan dengan algoritma deteksi objek lainnya.",
        "YOLOv8n (nano) untuk mendeteksi satu kelas objek utama: ikan mujair. Tidak dilakukan perbandingan dengan algoritma deteksi objek lainnya."
    ),
    (
        "Fokus parameter yang diamati adalah perilaku agregasi (berkumpulnya ikan) dan estimasi sisa pakan secara visual.",
        "Fokus parameter yang diamati adalah perilaku agregasi (berkumpulnya ikan) serta distribusi spasial ikan di dalam akuarium."
    ),

    # ── RUMUSAN MASALAH ──
    (
        "Bagaimana performa akurasi dan kecepatan algoritma YOLOv8n dalam mendeteksi objek ikan mujair dan sisa pakan secara real-time pada media akuarium?",
        "Bagaimana performa akurasi dan kecepatan algoritma YOLOv8n dalam mendeteksi objek ikan mujair secara real-time pada media akuarium?"
    ),
    (
        "Bagaimana merancang logika penentuan tingkat nafsu makan ikan berdasarkan parameter kepadatan agregasi dan jumlah sisa pakan yang terdeteksi?",
        "Bagaimana merancang logika penentuan tingkat nafsu makan ikan berdasarkan parameter kepadatan dan distribusi agregasi ikan?"
    ),

    # ── TUJUAN PENELITIAN ──
    (
        "Mengimplementasikan dan menguji performa algoritma YOLOv8n dalam mendeteksi objek ikan dan sisa pakan pada lingkungan akuarium",
        "Mengimplementasikan dan menguji performa algoritma YOLOv8n dalam mendeteksi objek ikan pada lingkungan akuarium"
    ),
    (
        "Merancang algoritma yang mampu mengintegrasikan data perilaku agregasi ikan (kepadatan/pergerakan) dan estimasi jumlah sisa pakan untuk menghasilkan status tingkat nafsu makan secara kuantitatif (Lapar dan Kenyang).",
        "Merancang algoritma yang mampu menganalisis data perilaku agregasi ikan (kepadatan dan distribusi spasial) untuk menghasilkan status tingkat nafsu makan secara kuantitatif (Lapar dan Kenyang)."
    ),

    # ── MANFAAT TEORETIS ──
    (
        "hubungan antara pola pengelompokan ikan dan keberadaan sisa pakan sebagai indikator yang dapat digunakan untuk menentukan tingkat kekenyangan ikan.",
        "hubungan antara pola pengelompokan dan distribusi ikan sebagai indikator yang dapat digunakan untuk menentukan tingkat kekenyangan ikan."
    ),

    # ── BAB 2 — Arsitektur YOLOv8 (Neck) ──
    (
        "sangat krusial untuk mendeteksi objek kecil seperti butiran pakan agar tidak hilang informasinya saat proses downsampling.",
        "sangat krusial untuk mendeteksi objek pada berbagai skala, termasuk ikan yang saling tumpang tindih dalam kondisi agregasi padat, agar tidak hilang informasinya saat proses downsampling."
    ),

    # ── BAB 3 — PENDEKATAN PENELITIAN ──
    (
        "dalam mendeteksi perilaku ikan dan sisa pakan.",
        "dalam mendeteksi perilaku ikan."
    ),

    # ── BAB 3 — STUDI LITERATUR ──
    (
        "perilaku ikan mujair, karakteristik sisa pakan, dan arsitektur teknis algoritma YOLOv8.",
        "perilaku ikan mujair dan arsitektur teknis algoritma YOLOv8."
    ),

    # ── BAB 3 — HIPOTESIS ──
    (
        "mampu mendeteksi objek ikan mujair dan sisa pakan pada kondisi pencahayaan alami",
        "mampu mendeteksi objek ikan mujair pada kondisi pencahayaan alami"
    ),
    (
        "Integrasi parameter perilaku agregasi dan deteksi sisa pakan mampu memberikan kesimpulan status nafsu makan",
        "Analisis parameter perilaku agregasi dan distribusi spasial ikan mampu memberikan kesimpulan status nafsu makan"
    ),

    # ── BAB 3 — SUMBER DATA ──
    (
        "hanya mencakup gambar dengan objek target berupa ikan dan pakan",
        "hanya mencakup gambar dengan objek target berupa ikan mujair"
    ),

    # ── BAB 3 — PENGUMPULAN DATA ──
    (
        "ketika sisa pakan mulai terlihat dan ikan cenderung menyebar.",
        "ketika ikan mulai menunjukkan perilaku dispersal dan menyebar ke seluruh area akuarium."
    ),
    (
        "Label yang digunakan terdiri dari dua kelas, yaitu ikan untuk objek tubuh ikan mujair dan pakan untuk butiran pakan yang mengapung di permukaan air, dengan format anotasi disimpan dalam berkas .txt sesuai standar YOLO.",
        "Label yang digunakan terdiri dari satu kelas, yaitu ikan (fish) untuk objek tubuh ikan mujair, dengan format anotasi disimpan dalam berkas .txt sesuai standar YOLO."
    ),

    # ── BAB 3 — CONFUSION MATRIX ──
    (
        "true positive (TP) yang menunjukkan kondisi ketika sistem berhasil mendeteksi objek ikan atau pakan sesuai dengan fakta di lapangan",
        "true positive (TP) yang menunjukkan kondisi ketika sistem berhasil mendeteksi objek ikan sesuai dengan fakta di lapangan"
    ),
    (
        "misalnya ketika sistem mengidentifikasi objek non-target seperti gelembung air sebagai pakan",
        "misalnya ketika sistem mengidentifikasi objek non-target seperti refleksi cahaya atau bayangan sebagai ikan"
    ),

    # ── COVER / JUDUL HALAMAN SAMPUL (versi CAPSLOCK) ──
    (
        "BERBASIS ANALISIS PERILAKU AGREGASI DAN SISA PAKAN MENGGUNAKAN YOLOV8",
        "BERBASIS ANALISIS PERILAKU AGREGASI MENGGUNAKAN YOLOV8N"
    ),

    # ── DAFTAR ISI — Sub-bab sisa pakan ──
    (
        "Penelitian Terkait Deteksi Sisa Pakan",
        "Penelitian Terkait Model Deteksi Ringan"
    ),

    # ── BAB 2 — TINJAUAN PUSTAKA: Deteksi Sisa Pakan ──
    (
        "sistem cerdas disarankan tidak hanya bergantung pada satu parameter (pakan saja), melainkan harus mengintegrasikan data perilaku respons",
        "sistem cerdas disarankan mengintegrasikan data perilaku respons"
    ),
    (
        "yang menggabungkan deteksi perilaku agregasi dan sisa pakan secara simultan dalam satu pipeline deteksi ringan.",
        "yang memanfaatkan deteksi perilaku agregasi dalam satu pipeline deteksi ringan."
    ),
    (
        "yang mengintegrasikan analisis perilaku agregasi dan sisa pakan menggunakan model YOLOv8n (Nano).",
        "yang menganalisis perilaku agregasi menggunakan model YOLOv8n (Nano)."
    ),

    # ── BAB 3 — PROSEDUR: mendokumentasikan kondisi sisa pakan ──
    (
        "serta mendokumentasikan kondisi sisa pakan.",
        "serta mendokumentasikan kondisi distribusi ikan."
    ),
    (
        "kondisi permukaan air yang mengandung sisa pakan maupun tidak.",
        "berbagai kondisi permukaan air."
    ),

    # ── BAB 3 — VARIABEL OPERASIONAL: Sisa Pakan ──
    (
        "Sisa Pakan (Uneaten Feed)",
        "Distribusi Spasial Ikan"
    ),
    (
        "Didefinisikan sebagai objek butiran kecil yang terklasifikasi sebagai kelas 'pakan' yang mengapung di permukaan air.",
        "Didefinisikan sebagai pola sebaran posisi ikan yang terdeteksi dalam frame."
    ),
    (
        "Sisa pakan dihitung berdasarkan jumlah bounding box pakan yang terdeteksi secara konsisten selama kurun waktu tertentu setelah penebaran pakan.",
        "Distribusi spasial dihitung berdasarkan jarak rata-rata antar bounding box ikan yang terdeteksi secara konsisten selama kurun waktu tertentu."
    ),

    # ── BAB 3 — LOGIKA PENENTUAN STATUS ──
    (
        "ditentukan berdasarkan korelasi antara agregasi dan sisa pakan",
        "ditentukan berdasarkan analisis pola agregasi ikan"
    ),
    (
        "DAN sisa pakan rendah/nol (Low Feed Residue).",
        "dan ikan berkumpul padat di permukaan."
    ),
    (
        "DAN sisa pakan mulai terdeteksi banyak (High Feed Residue).",
        "dan ikan menyebar ke seluruh area akuarium."
    ),
    (
        "jumlah objek sisa pakan yang terdeteksi pada setiap frame. Kedua variabel tersebut",
        "jarak rata-rata antar ikan yang terdeteksi pada setiap frame. Kedua variabel tersebut"
    ),

    # ── BAB 3 — LAMPIRAN: Tabel Definisi Kelas ──
    (
        "pakan (pellet)",
        "[DIHAPUS]"
    ),
    (
        "Butiran pakan terapung yang belum dimakan oleh ikan.",
        "[Kelas ini telah dihapus dari dataset]"
    ),

    # ── MANFAAT PRAKTIS ──
    (
        "serta mengurangi pencemaran air akibat sisa pakan.",
        "serta mengurangi pencemaran air akibat pakan berlebih."
    ),

    # ── FORMULA ──
    (
        "ambang batas sisa pakan",
        "ambang batas distribusi"
    ),
]


def replace_in_paragraph(paragraph, old_text, new_text):
    """
    Ganti teks dalam paragraph sambil mempertahankan formatting.
    
    Strategi: gabungkan semua run text, cari pattern, lalu rebuild runs.
    """
    # Gabungkan teks dari semua runs
    full_text = "".join(run.text for run in paragraph.runs)
    
    if old_text not in full_text:
        return False
    
    # Lakukan penggantian
    new_full_text = full_text.replace(old_text, new_text)
    
    if not paragraph.runs:
        return False
    
    # Simpan formatting dari run pertama yang punya teks
    # Lalu taruh semua teks baru di run pertama, kosongkan sisanya
    first_run = paragraph.runs[0]
    first_run.text = new_full_text
    
    for run in paragraph.runs[1:]:
        run.text = ""
    
    return True


def replace_in_tables(doc, old_text, new_text):
    """Cari dan ganti di semua tabel."""
    count = 0
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    if replace_in_paragraph(paragraph, old_text, new_text):
                        count += 1
    return count


def main():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Skripsi (1).docx")
    dst = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Skripsi (REVISI v2).docx")
    
    if not os.path.isfile(src):
        print(f"[ERROR] File tidak ditemukan: {src}")
        sys.exit(1)
    
    print(f"[*] Membuka: {src}")
    doc = Document(src)
    
    print(f"[*] Memulai {len(REPLACEMENTS)} penggantian teks...\n")
    
    total_replaced = 0
    
    for i, (old, new) in enumerate(REPLACEMENTS, 1):
        found = 0
        
        # Cari di paragraf biasa
        for para in doc.paragraphs:
            if replace_in_paragraph(para, old, new):
                found += 1
        
        # Cari di tabel
        found += replace_in_tables(doc, old, new)
        
        # Cari di header/footer
        for section in doc.sections:
            for header in [section.header, section.first_page_header, section.even_page_header]:
                if header and header.is_linked_to_previous is False:
                    for para in header.paragraphs:
                        if replace_in_paragraph(para, old, new):
                            found += 1
            for footer in [section.footer, section.first_page_footer, section.even_page_footer]:
                if footer and footer.is_linked_to_previous is False:
                    for para in footer.paragraphs:
                        if replace_in_paragraph(para, old, new):
                            found += 1
        
        status = "OK" if found > 0 else "!! TIDAK DITEMUKAN"
        # Tampilkan 60 karakter pertama dari teks lama
        preview = old[:80].replace("\n", " ")
        print(f"  [{i:2d}/{len(REPLACEMENTS)}] {status} ({found}x) - \"{preview}...\"")
        total_replaced += found
    
    # Juga cari-ganti teks pendek generik yang tersisa
    extra_short = [
        ("ikan dan sisa pakan", "ikan"),
        ("ikan atau pakan", "ikan"),
        ("ikan dan pakan", "ikan mujair"),
        ("ikan maupun pakan", "ikan"),
    ]
    extra_count = 0
    for old, new in extra_short:
        for para in doc.paragraphs:
            if replace_in_paragraph(para, old, new):
                extra_count += 1
        extra_count += replace_in_tables(doc, old, new)
    if extra_count > 0:
        print(f"\n  [EXTRA] {extra_count} penggantian teks pendek tambahan")
        total_replaced += extra_count
    
    print(f"\n[*] Total penggantian berhasil: {total_replaced}")
    print(f"[*] Menyimpan ke: {dst}")
    doc.save(dst)
    print(f"[OK] Selesai! File revisi: {dst}")
    print("")
    print("[!] PENTING: Periksa file hasil revisi secara manual, terutama:")
    print("    - Tabel di Lampiran (Definisi Kelas, Confusion Matrix, Metrik)")
    print("    - Formatting (bold, italic) mungkin perlu disesuaikan ulang")


if __name__ == "__main__":
    main()
