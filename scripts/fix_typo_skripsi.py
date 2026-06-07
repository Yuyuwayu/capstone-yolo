import os
import sys
from docx import Document

REPLACEMENTS = [
    # 1. Typo Judul Proposal
    (
        "Sistem deteksi tingkat nafsu makan ikan mujair berbasis analisis perilaku agregasi menggunakan YOLOv8n",
        "Sistem Deteksi Tingkat Nafsu Makan Ikan Mujair Berbasis Analisis Perilaku Agregasi Menggunakan YOLOv8n"
    ),
    # 2. Typo Software
    (
        "Spesifikasi Softawre",
        "Spesifikasi Software"
    ),
    # 3. Kontradiksi Latar Belakang (Hu et al & sisa pakan)
    (
        "Penelitian mereka membuktikan bahwa deteksi visual yang akurat terhadap sisa pakan adalah indikator paling objektif untuk menghentikan pemberian pakan tepat waktu, sehingga mencegah lonjakan kadar nitrogen yang berbahaya bagi ekosistem tambak",
        "Meskipun deteksi sisa pakan terbukti objektif, implementasinya di lapangan sering terkendala oleh resolusi kamera dan butiran pakan yang sangat kecil. Oleh karena itu, diperlukan pendekatan alternatif berbasis fitur makro, yaitu analisis perilaku agregasi ikan yang lebih aplikatif dan komputasional ringan tanpa memerlukan deteksi sisa pakan secara mikroskopis"
    ),
    # 4. Kontradiksi Kolam vs Akuarium
    (
        "citra digital kolam ikan mujair",
        "citra digital akuarium ikan mujair"
    ),
    # 5. Typo Tanda Baca
    (
        "akuarium?.",
        "akuarium?"
    ),
    (
        "agregasi ikan?.",
        "agregasi ikan?"
    ),
    (
        "(ground truth) )",
        "(ground truth)"
    ),
    # 6. Koreksi Jarak Rata-rata Bounding Box
    (
        "Distribusi spasial dihitung berdasarkan jarak rata-rata antar bounding box ikan yang terdeteksi secara konsisten selama kurun waktu tertentu.",
        "Distribusi spasial dihitung berdasarkan tingkat kepadatan dan penyebaran sekumpulan bounding box ikan yang terdeteksi secara konsisten selama kurun waktu tertentu."
    )
]

def replace_in_paragraph(paragraph, old_text, new_text):
    full_text = "".join(run.text for run in paragraph.runs)
    if old_text not in full_text:
        return False
    
    new_full_text = full_text.replace(old_text, new_text)
    if not paragraph.runs:
        return False
    
    paragraph.runs[0].text = new_full_text
    for run in paragraph.runs[1:]:
        run.text = ""
    return True

def replace_in_tables(doc, old_text, new_text):
    count = 0
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    if replace_in_paragraph(paragraph, old_text, new_text):
                        count += 1
    return count

def main():
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Skripsi (REVISI v2).docx")
    dst = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Skripsi (FINAL).docx")
    
    if not os.path.isfile(src):
        print(f"[ERROR] File tidak ditemukan: {src}")
        sys.exit(1)
    
    print(f"[*] Membuka: {src}")
    doc = Document(src)
    total_replaced = 0
    
    for i, (old, new) in enumerate(REPLACEMENTS, 1):
        found = 0
        for para in doc.paragraphs:
            if replace_in_paragraph(para, old, new):
                found += 1
        found += replace_in_tables(doc, old, new)
        
        # Check headers/footers
        for section in doc.sections:
            for header in [section.header, section.first_page_header, section.even_page_header]:
                if header and not header.is_linked_to_previous:
                    for para in header.paragraphs:
                        if replace_in_paragraph(para, old, new):
                            found += 1
        
        status = "OK" if found > 0 else "FAIL"
        print(f"  [{i}/{len(REPLACEMENTS)}] {status} ({found}x) - {old[:30]}...")
        total_replaced += found

    print(f"\n[*] Total perbaikan: {total_replaced}")
    doc.save(dst)
    print(f"[OK] Disimpan sebagai: {dst}")

if __name__ == "__main__":
    main()
