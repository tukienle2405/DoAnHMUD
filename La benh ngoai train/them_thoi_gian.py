
from pathlib import Path

ROOT = Path(r"D:\DoanHMvUD\La benh ngoai train")

SOURCE = ROOT / "danh_gia_plantdoc_1194.py"
OUTPUT = ROOT / "danh_gia_plantdoc_1194_co_thoi_gian.py"

if not SOURCE.exists():
    raise FileNotFoundError(f"Khong tim thay: {SOURCE}")

code = SOURCE.read_text(encoding="utf-8-sig")


def replace_once(old, new, description):
    global code

    count = code.count(old)

    if count != 1:
        raise ValueError(
            f"Khong tim thay dung vi tri: {description}. "
            f"So lan xuat hien: {count}"
        )

    code = code.replace(old, new, 1)


# 1. Bổ sung thư viện thời gian
replace_once(
    "import argparse\n",
    "import argparse\nimport time\n"
    "from datetime import datetime, timedelta\n",
    "Import thoi gian"
)


# 2. Hàm định dạng giây thành giờ:phút:giây
replace_once(
    'EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}',
    '''EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def fmt_time(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
'''.rstrip(),
    "Ham dinh dang thoi gian"
)


# 3. Ghi nhận thời gian bắt đầu
replace_once(
    '    print("Thiet bi:", device)\n',
    '''    print("Thiet bi:", device)

    bat_dau = time.perf_counter()

    print(
        "Bat dau:",
        datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        flush=True
    )
''',
    "Thoi gian bat dau"
)


# 4. Bắt đầu đo thời gian xử lý ảnh
replace_once(
    '''    results = []

    with torch.inference_mode():''',
    '''    results = []

    xu_ly_bat_dau = time.perf_counter()
    tong_anh = len(records)

    with torch.inference_mode():''',
    "Bat dau xu ly anh"
)


# 5. Thay thông báo tiến độ cũ bằng tiến độ + ETA
replace_once(
    '''            print(
                f"Da xu ly: "
                f"{min(start + len(batch), len(records))}"
                f"/{len(records)}",
                flush=True
            )''',
    '''            da_xu_ly = min(start + len(batch), tong_anh)

            da_chay = time.perf_counter() - xu_ly_bat_dau

            toc_do = (
                da_xu_ly / da_chay
                if da_chay > 0 else 0.0
            )

            anh_con_lai = tong_anh - da_xu_ly

            eta = (
                anh_con_lai / toc_do
                if toc_do > 0 else 0.0
            )

            gio_du_kien = (
                datetime.now() + timedelta(seconds=eta)
            )

            phan_tram = 100 * da_xu_ly / tong_anh

            print(
                f"[{phan_tram:5.1f}%] "
                f"{da_xu_ly}/{tong_anh} anh | "
                f"Da chay: {fmt_time(da_chay)} | "
                f"Con lai: ~{fmt_time(eta)} | "
                f"Xong luc: {gio_du_kien:%H:%M:%S} | "
                f"{toc_do:.2f} anh/s",
                flush=True
            )''',
    "Tien do va ETA"
)


# 6. Tổng thời gian sau khi xuất báo cáo
replace_once(
    '    print("\\nDa xuat bao cao tai:", OUTPUT)',
    '''    print("\\nDa xuat bao cao tai:", OUTPUT)

    tong_thoi_gian = time.perf_counter() - bat_dau

    print("\\n========== THOI GIAN ==========")
    print(
        "Ket thuc:",
        datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    )
    print("Tong thoi gian:", fmt_time(tong_thoi_gian))
''',
    "Tong thoi gian"
)


# Lưu thành file mới, giữ nguyên file cũ
OUTPUT.write_text(code, encoding="utf-8")

print("DA THEM THOI GIAN THANH CONG!")
print("File moi:", OUTPUT)
print("File goc khong thay doi.")
