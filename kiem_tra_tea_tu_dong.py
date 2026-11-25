
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import hashlib
import csv

ROOT = Path(r"D:\DoanHMvUD")
DATA = ROOT / "PlantVillage" / "Tea Leaf Dataset"
OUT = ROOT / "dataset_review" / "tea_review"
OUT.mkdir(parents=True, exist_ok=True)

EXT = {".jpg", ".jpeg", ".png", ".bmp",
       ".webp", ".tif", ".tiff"}


def save_csv(filename, header, rows):
    with open(OUT / filename, "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def dhash_file(path):
    with Image.open(path) as img:
        img = img.convert("L").resize(
            (9, 8), Image.Resampling.LANCZOS
        )
        pixels = img.tobytes()

    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (
                pixels[y * 9 + x] >
                pixels[y * 9 + x + 1]
            )
    return value


if not DATA.is_dir():
    raise FileNotFoundError(DATA)

records = []
errors = []

print("Dang quet dataset Che...")

for path in DATA.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix.lower() not in EXT:
        continue

    parts = path.relative_to(DATA).parts

    if "Diseased Leaves" in parts:
        cls = path.parent.name
    elif "Healthy Leaves" in parts:
        cls = "Healthy"
    else:
        cls = "UNMAPPED"

    try:
        records.append({
            "class": cls,
            "path": str(path),
            "sha": hash_file(path),
            "dhash": dhash_file(path)
        })
    except Exception as e:
        errors.append([str(path), str(e)])

    if (len(records) + len(errors)) % 5000 == 0:
        print("Da kiem tra:",
              len(records) + len(errors))

by_sha = defaultdict(list)
by_dhash = defaultdict(list)

for item in records:
    by_sha[item["sha"]].append(item)
    by_dhash[item["dhash"]].append(item)

conflicts = set()
duplicates = set()
cross_rows = []
same_rows = []

for sha, items in by_sha.items():
    classes = {x["class"] for x in items}

    if len(classes) > 1:
        for x in items:
            conflicts.add(x["path"])
            cross_rows.append([
                sha, x["class"], x["path"]
            ])

    elif len(items) > 1:
        items.sort(key=lambda x: x["path"])

        for x in items[1:]:
            duplicates.add(x["path"])
            same_rows.append([
                sha, x["class"], x["path"]
            ])

# dHash bang nhau chi la dau hieu nghi van.
# Khong tu dong xoa dua vao dHash.
visual_rows = []
visual_groups = 0

for value, items in by_dhash.items():
    classes = {x["class"] for x in items}
    shas = {x["sha"] for x in items}

    if len(classes) < 2 or len(shas) < 2:
        continue

    visual_groups += 1

    # Luu toi da 2 anh mau cho moi lop trong nhom
    samples = defaultdict(list)
    for x in items:
        if len(samples[x["class"]]) < 2:
            samples[x["class"]].append(x["path"])

    for cls, paths in samples.items():
        for path in paths:
            visual_rows.append([
                format(value, "016x"),
                cls,
                len(items),
                path
            ])

status_rows = []
status_counts = Counter()
class_counts = Counter()

for x in records:
    cls, path = x["class"], x["path"]

    if path in conflicts:
        status = "CACH_LY_KHAC_NHAN"
    elif path in duplicates:
        status = "BO_BAN_SAO_CUNG_LOP"
    elif cls == "UNMAPPED":
        status = "CAN_ANH_XA_NHAN"
    else:
        status = "CHUA_XAC_MINH_NGUON"

    status_counts[status] += 1
    class_counts[cls] += 1

    status_rows.append([
        path, cls, x["sha"], status
    ])

save_csv(
    "tea_toan_bo.csv",
    ["path", "class", "sha256", "status"],
    status_rows
)

save_csv(
    "tea_trung_sha_khac_lop.csv",
    ["sha256", "class", "path"],
    cross_rows
)

save_csv(
    "tea_trung_sha_cung_lop.csv",
    ["sha256", "class", "path"],
    same_rows
)

save_csv(
    "tea_nghi_giong_dhash.csv",
    ["dhash", "class", "group_size", "path"],
    visual_rows
)

save_csv(
    "tea_anh_loi.csv",
    ["path", "error"],
    errors
)

report = [
    "BAO CAO KIEM TRA DATASET CHE",
    "=" * 45,
    f"So anh doc duoc: {len(records)}",
    f"So anh loi: {len(errors)}",
    f"So lop tim thay: {len(class_counts)}",
    "",
    "SO ANH THEO LOP:"
]

for cls, count in sorted(class_counts.items()):
    report.append(f"{cls}: {count}")

report.extend([
    "",
    "KET QUA KIEM TRA:"
])

for status, count in sorted(status_counts.items()):
    report.append(f"{status}: {count}")

report.extend([
    f"So nhom dHash nghi van: {visual_groups}",
    "",
    "Chua xac minh nguon augmentation.",
    "Chua kiem tra anh gan trung co dHash khac nhau.",
    "Chua chia Train/Validation/Test.",
    "Khong xoa hoac di chuyen anh goc."
])

result = "\n".join(report)
(OUT / "bao_cao_tea.txt").write_text(
    result, encoding="utf-8"
)

print("\n" + result)
print("\nDa luu tai:", OUT)
