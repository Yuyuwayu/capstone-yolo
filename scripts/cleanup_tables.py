import os
import sys
from docx import Document

def remove_row(table, row):
    tbl = table._tbl
    tr = row._tr
    tbl.remove(tr)

def main():
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Skripsi (FINAL).docx")
    dst_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Skripsi (FINAL v2).docx")
    
    if not os.path.isfile(src_path):
        print(f"[ERROR] File tidak ditemukan: {src_path}")
        sys.exit(1)
        
    doc = Document(src_path)
    removed_count = 0
    
    for table in doc.tables:
        rows_to_remove = []
        for row in table.rows:
            cells_text = [cell.text.strip().lower() for cell in row.cells]
            
            if "pakan" in cells_text:
                rows_to_remove.append(row)
            elif "[dihapus]" in cells_text:
                rows_to_remove.append(row)
            elif len(cells_text) > 1 and cells_text[1] == "" and cells_text[0] == "":
                rows_to_remove.append(row)

        for row in rows_to_remove:
            remove_row(table, row)
            removed_count += 1
            
    if removed_count > 0:
        doc.save(dst_path)
        print(f"[OK] Berhasil menghapus {removed_count} baris dari tabel-tabel di dokumen.")
        print(f"[OK] Disimpan sebagai: {dst_path}")
    else:
        print("[INFO] Tidak ada baris yang perlu dihapus.")

if __name__ == "__main__":
    main()
