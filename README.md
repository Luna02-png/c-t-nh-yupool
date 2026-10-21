HYBRID IMAGE FALLBACK PATCH - 2026-09-14

Mục tiêu:
- Giữ nguyên flow cũ và toàn bộ chức năng hiện có.
- Google Sheets vẫn thử insert ảnh trực tiếp như trước.
- Nếu insert thành công: giữ nguyên ảnh trên Sheet và khi export XLSX không nhúng trùng.
- Nếu ảnh đã tải lên TEMP Drive nhưng insert vào Google Sheets thất bại: log F/I ghi "⚠ CẦN EXPORT".
- Khi bấm "📤 Export Excel (.xlsx)": Python chỉ nhúng fallback cho các dòng có log "CẦN EXPORT" và chưa có ảnh trong XLSX.

File:
- Code.gs: full Apps Script.
- yupoo_runner.py: full Python runner, giữ các flow Yupoo/Store/Direct/Article/cache/retry hiện tại và thêm fallback XLSX.

Dependency mới (cài 1 lần trong .venv):
  python -m pip install openpyxl

Cách dùng:
1. Thay Code.gs bằng file trong ZIP.
2. Thay yupoo_runner.py bằng file trong ZIP.
3. Chạy: python yupoo_runner.py
4. Dùng Smart Batch / Update selected rows như cũ.
5. Dòng insert được: log OK như cũ.
6. Dòng insert không được nhưng đã có ảnh TEMP Drive: log "⚠ CẦN EXPORT".
7. Khi cần file Excel hoàn chỉnh, bấm "📤 Export Excel (.xlsx)".

Không đổi:
- Cấu trúc cột A-I.
- Link1 B -> ảnh E -> log F.
- Link2 G -> ảnh H -> log I.
- Smart Batch / Retry / Update selected rows / clear / trash / runner config / token / cache / TEMP folder / download logic.
