
import os
from pathlib import Path

# Đường dẫn dữ liệu
ROOT = Path(r"D:\DoanHMvUD\PlantVillage")

# File lưu kết quả
OUTPUT = Path(r"D:\DoanHMvUD\thong_ke_plantvillage.txt")

# Các định dạng ảnh
IMAGE_EXT = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}

if not ROOT.is_dir():
    raise FileNotFoundError(f"Không tìm thấy: {ROOT}")

# Lưu thông tin từng thư mục
data = {}

# Đọc toàn bộ cấu trúc thư mục một lần
for current, folders, files in os.walk(ROOT):
    folders.sort()

    path = Path(current)

    image_files = [
        f for f in files
        if Path(f).suffix.lower() in IMAGE_EXT
    ]

    data[path] = {
        "children": [path / f for f in folders],
        "files_direct": len(files),
        "images_direct": len(image_files),
        "files_total": len(files),
        "images_total": len(image_files)
    }

# Cộng số file từ các thư mục con lên thư mục cha
for path in sorted(
    data,
    key=lambda p: len(p.parts),
    reverse=True
):
    for child in data[path]["children"]:
        if child in data:
            data[path]["files_total"] += data[child]["files_total"]
            data[path]["images_total"] += data[child]["images_total"]

lines = []


def add_line(text):
    print(text)
    lines.append(text)


def show_folder(path, level=0):
    info = data[path]

    indent = "    " * level

    add_line(
        f"{indent}- {path.name}"
        f" | {info['images_total']:,} ảnh"
        f" | {info['files_total']:,} file"
        f" | {len(info['children'])} thư mục con"
    )

    # Nếu ảnh nằm trực tiếp trong thư mục
    if info["images_direct"] > 0:
        add_line(
            f"{indent}    Ảnh trực tiếp: "
            f"{info['images_direct']:,}"
        )

    # In các thư mục bên trong
    for child in info["children"]:
        if child in data:
            show_folder(child, level + 1)


add_line("=" * 65)
add_line("THỐNG KÊ TOÀN BỘ PLANTVILLAGE")
add_line("=" * 65)

add_line(
    f"Tổng số thư mục cấp 1: "
    f"{len(data[ROOT]['children'])}"
)

add_line(
    f"Tổng số ảnh: "
    f"{data[ROOT]['images_total']:,}"
)

add_line(
    f"Tổng số file: "
    f"{data[ROOT]['files_total']:,}"
)

add_line("\nDANH SÁCH THƯ MỤC:\n")

show_folder(ROOT)

# Xuất kết quả ra file TXT
OUTPUT.write_text(
    "\n".join(lines),
    encoding="utf-8-sig"
)

print("\nĐã lưu kết quả tại:")
print(OUTPUT)
