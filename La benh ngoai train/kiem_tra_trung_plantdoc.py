
from pathlib import Path
from collections import defaultdict
import hashlib
import csv

ROOT = Path(r"D:\DoanHMvUD")

# Ảnh PlantDoc vừa sao chép
NEW = ROOT / "external_field_pool_v2" / "PlantDoc_ung_vien"

# Dữ liệu gốc
OLD = ROOT / "PlantVillage"

OUT = ROOT / "external_field_pool_v2"
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

if not NEW.exists():
    raise FileNotFoundError(f"Khong tim thay: {NEW}")

if not OLD.exists():
    raise FileNotFoundError(f"Khong tim thay: {OLD}")

# Đọc toàn bộ ảnh PlantDoc
images = [
    p for p in NEW.rglob("*")
    if p.is_file() and p.suffix.lower() in EXT
]

print(f"Anh PlantDoc: {len(images)}")

# Lưu SHA256 theo kích thước để đối chiếu nhanh
lookup = defaultdict(lambda: defaultdict(list))

for p in images:
    lookup[p.stat().st_size][sha256(p)].append(p)

duplicates = set()
scanned = 0

# So sánh với toàn bộ PlantVillage
for p in OLD.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in EXT:
        continue

    scanned += 1

    if scanned % 10000 == 0:
        print(f"Da quet {scanned} anh goc")

    size = p.stat().st_size

    # Chỉ tính hash khi kích thước trùng
    if size not in lookup:
        continue

    digest = sha256(p)

    for match in lookup[size].get(digest, []):
        duplicates.add(match)

# Thống kê theo từng nhóm ảnh
groups = defaultdict(list)

for p in images:
    groups[p.parent.name].append(p)

report = []

for group, files in sorted(groups.items()):
    dup = sum(p in duplicates for p in files)

    report.append({
        "Nhom_anh": group,
        "Tong_anh": len(files),
        "Trung_PlantVillage": dup,
        "Khong_trung_SHA256": len(files) - dup
    })

    print(
        f"{group}: {len(files)} anh, "
        f"{dup} trung byte"
    )

# Xuất báo cáo
file = OUT / "BAO_CAO_PLANTDOC_KIEM_TRA.csv"

with open(file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(report[0].keys())
    )
    writer.writeheader()
    writer.writerows(report)

# Xuất danh sách ảnh trùng
duplicate_file = OUT / "DANH_SACH_ANH_TRUNG_PLANTDOC.txt"
duplicate_file.write_text(
    "\n".join(str(p) for p in sorted(duplicates)),
    encoding="utf-8"
)

print("\n========== KET QUA ==========")
print("Tong anh:", len(images))
print("Anh trung SHA256:", len(duplicates))
print("Khong trung SHA256:", len(images) - len(duplicates))
print("Bao cao:", file)
print("Danh sach anh trung:", duplicate_file)

from pathlib import Path
from collections import defaultdict
import hashlib
import csv

ROOT = Path(r"D:\DoanHMvUD")

# Ảnh PlantDoc vừa sao chép
NEW = ROOT / "external_field_pool_v2" / "PlantDoc_ung_vien"

# Dữ liệu gốc
OLD = ROOT / "PlantVillage"

OUT = ROOT / "external_field_pool_v2"
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()

if not NEW.exists():
    raise FileNotFoundError(f"Khong tim thay: {NEW}")

if not OLD.exists():
    raise FileNotFoundError(f"Khong tim thay: {OLD}")

# Đọc toàn bộ ảnh PlantDoc
images = [
    p for p in NEW.rglob("*")
    if p.is_file() and p.suffix.lower() in EXT
]

print(f"Anh PlantDoc: {len(images)}")

# Lưu SHA256 theo kích thước để đối chiếu nhanh
lookup = defaultdict(lambda: defaultdict(list))

for p in images:
    lookup[p.stat().st_size][sha256(p)].append(p)

duplicates = set()
scanned = 0

# So sánh với toàn bộ PlantVillage
for p in OLD.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in EXT:
        continue

    scanned += 1

    if scanned % 10000 == 0:
        print(f"Da quet {scanned} anh goc")

    size = p.stat().st_size

    # Chỉ tính hash khi kích thước trùng
    if size not in lookup:
        continue

    digest = sha256(p)

    for match in lookup[size].get(digest, []):
        duplicates.add(match)

# Thống kê theo từng nhóm ảnh
groups = defaultdict(list)

for p in images:
    groups[p.parent.name].append(p)

report = []

for group, files in sorted(groups.items()):
    dup = sum(p in duplicates for p in files)

    report.append({
        "Nhom_anh": group,
        "Tong_anh": len(files),
        "Trung_PlantVillage": dup,
        "Khong_trung_SHA256": len(files) - dup
    })

    print(
        f"{group}: {len(files)} anh, "
        f"{dup} trung byte"
    )

# Xuất báo cáo
file = OUT / "BAO_CAO_PLANTDOC_KIEM_TRA.csv"

with open(file, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(report[0].keys())
    )
    writer.writeheader()
    writer.writerows(report)

# Xuất danh sách ảnh trùng
duplicate_file = OUT / "DANH_SACH_ANH_TRUNG_PLANTDOC.txt"
duplicate_file.write_text(
    "\n".join(str(p) for p in sorted(duplicates)),
    encoding="utf-8"
)

print("\n========== KET QUA ==========")
print("Tong anh:", len(images))
print("Anh trung SHA256:", len(duplicates))
print("Khong trung SHA256:", len(images) - len(duplicates))
print("Bao cao:", file)
print("Danh sach anh trung:", duplicate_file)
