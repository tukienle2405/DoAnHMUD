
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import csv

ROOT = Path(r"D:\DoanHMvUD")

BASE = ROOT / "dataset_review" / "dataset_97_lop"
DUP = BASE / "kiem_tra_trung"
OUT = BASE / "du_lieu_sau_lam_sach"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = BASE / "dataset_manifest_SO_BO.csv"
DETAIL = DUP / "chi_tiet_anh_trung.csv"

for file in (MANIFEST, DETAIL):
    if not file.is_file():
        raise FileNotFoundError(file)

def save_csv(path, header, rows):
    with open(path, "w", encoding="utf-8-sig",
              newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

# 1. Doc ket qua SHA da tinh
groups = defaultdict(list)

with open(DETAIL, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        groups[row["sha256"]].append(row)

conflicts = set()
duplicates = set()

for sha, items in groups.items():
    issues = {x["issue"] for x in items}
    labels = {x["class"] for x in items}

    if len(issues) != 1:
        raise ValueError("Trang thai SHA khong nhat quan: " + sha)

    if "MAU_THUAN_NHAN" in issues or len(labels) > 1:
        # Khong tu doan nhan nao dung
        conflicts.update(x["path"] for x in items)
    else:
        # Giu 1 ban dai dien cho moi nhom trung cung nhan
        paths = sorted({x["path"] for x in items})
        duplicates.update(paths[1:])

print("Anh mau thuan nhan:", len(conflicts))
print("Ban sao du kien bo qua:", len(duplicates))

# 2. Doc va kiem tra anh trong manifest
OUTPUT = OUT / "dataset_manifest_SAU_SHA.csv"
REJECTED = OUT / "anh_tam_loai.csv"

fields = [
    "path", "class", "split",
    "status", "source", "group_id"
]

counts = Counter()
class_counts = Counter()
original_classes = set()
processed = 0

with (
    open(MANIFEST, encoding="utf-8-sig", newline="") as source,
    open(OUTPUT, "w", encoding="utf-8-sig", newline="") as target,
    open(REJECTED, "w", encoding="utf-8-sig", newline="") as badfile
):
    reader = csv.DictReader(source)

    writer = csv.DictWriter(target, fieldnames=fields)
    writer.writeheader()

    bad_writer = csv.writer(badfile)
    bad_writer.writerow(["path", "class", "reason"])

    for row in reader:
        processed += 1
        path = Path(row["path"])
        name = row["class"]
        original_classes.add(name)

        reason = None

        try:
            if not path.is_file():
                reason = "THIEU_FILE"
            elif path.stat().st_size == 0:
                reason = "FILE_RONG"
            elif str(path) in conflicts:
                reason = "MAU_THUAN_NHAN"
            elif str(path) in duplicates:
                reason = "BAN_SAO_CUNG_NHAN"
            else:
                # Kiem tra co doc duoc du lieu anh that khong
                with Image.open(path) as img:
                    img.load()

                    if img.width < 32 or img.height < 32:
                        reason = "ANH_QUA_NHO"

        except Exception as exc:
            reason = "ANH_LOI: " + str(exc)[:150]

        if reason is not None:
            counts[reason.split(":")[0]] += 1
            bad_writer.writerow([str(path), name, reason])
        else:
            writer.writerow({
                key: row[key] for key in fields
            })
            counts["GIU_LAI"] += 1
            class_counts[name] += 1

        if processed % 10000 == 0:
            print(
                f"Da kiem tra {processed:,} anh | "
                f"Giu lai {counts['GIU_LAI']:,}"
            )

# 3. Kiem tra so lop sau loc
missing_classes = sorted(
    original_classes - set(class_counts)
)

save_csv(
    OUT / "tong_hop_sau_SHA.csv",
    ["class", "images"],
    [
        [name, class_counts.get(name, 0)]
        for name in sorted(original_classes)
    ]
)

report = [
    "BAO CAO LAM SACH DATASET 97 LOP",
    "=" * 50,
    f"Tong anh dau vao: {processed:,}",
    f"So lop dau vao: {len(original_classes)}",
    f"So lop con anh: {len(class_counts)}",
    "",
    "KET QUA:"
]

for reason, count in sorted(counts.items()):
    report.append(f"{reason}: {count:,}")

report.extend([
    "",
    f"So lop het anh: {len(missing_classes)}",
    *missing_classes,
    "",
    "LUU Y:",
    "Khong xoa hay di chuyen anh goc.",
    "Chua kiem tra anh gan trung tren toan bo dataset.",
    "Chua xac minh nhan benh bang chuyen gia.",
    "Giu nguyen split pending chua duoc chia.",
    "Chua san sang train mo hinh 97 lop."
])

result = "\n".join(report)

(OUT / "bao_cao_sau_SHA.txt").write_text(
    result, encoding="utf-8"
)

print("\n" + result)
print("\nDa luu tai:", OUT)
