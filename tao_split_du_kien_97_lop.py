
from pathlib import Path
from collections import defaultdict, Counter
import csv
import random

ROOT = Path(r"D:\DoanHMvUD")

BASE = ROOT / "dataset_review" / "dataset_97_lop"
INPUT = (
    BASE / "kiem_tra_gan_trung"
    / "dataset_manifest_NHOM_SO_BO.csv"
)

OUT = BASE / "split_97_lop"
OUT.mkdir(parents=True, exist_ok=True)

OUTPUT = OUT / "dataset_manifest_SPLIT_DU_KIEN.csv"
REPORT = OUT / "bao_cao_split.txt"

SEED = 42
VALID_SPLITS = {"train", "val", "test"}
REVIEW_FLAGS = {
    "CAN_XEM_KHAC_NHAN",
    "CAN_XEM_BUCKET_DONG",
    "LOI_ANH"
}

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

with open(INPUT, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

print("Tong anh:", len(rows))

# 1. Gom anh pending theo lop va nhom gan trung
groups_by_class = defaultdict(lambda: defaultdict(list))
quarantined = []

for index, row in enumerate(rows):
    if row["split"] != "pending":
        continue

    if row["near_review"] in REVIEW_FLAGS:
        row["split"] = "exclude_review"
        quarantined.append(row)
        continue

    label = row["class"]
    near_group = row["near_group_id"]

    # Anh khong co near_group_id duoc giu rieng
    if not near_group:
        near_group = f"SINGLE_{index}"

    key = (label, near_group)
    groups_by_class[label][key].append(index)

# 2. Chia theo nhom, khong chia tung anh
rng = random.Random(SEED)

for label in sorted(groups_by_class):
    class_groups = groups_by_class[label]

    keys = sorted(class_groups.keys())
    rng.shuffle(keys)

    n = len(keys)

    if n < 3:
        raise ValueError(
            f"Lop {label} chi co {n} nhom, "
            "khong du de chia 3 tap"
        )

    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    # Dam bao moi tap co it nhat 1 nhom
    n_train = max(1, n_train)
    n_val = max(1, n_val)

    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1

    for position, key in enumerate(keys):
        if position < n_train:
            split = "train"
        elif position < n_train + n_val:
            split = "val"
        else:
            split = "test"

        for index in class_groups[key]:
            rows[index]["split"] = split

# 3. Kiem tra roi ri theo near_group_id
group_splits = defaultdict(set)

for row in rows:
    if row["split"] not in VALID_SPLITS:
        continue

    group = row.get("near_group_id", "").strip()

    if row["source"] == "TEA_GROUPED":
        group = row["group_id"]

    if not group:
        continue

    key = (row["class"], group)
    group_splits[key].add(row["split"])

leakage = [
    (key, splits)
    for key, splits in group_splits.items()
    if len(splits) > 1
]

if leakage:
    raise RuntimeError(
        f"Phat hien {len(leakage)} nhom "
        "nam trong nhieu tap!"
    )

# 4. Ghi manifest
fields = list(rows[0].keys()) + ["needs_label_review"]

with open(
    OUTPUT, "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for row in rows:
        output_row = dict(row)

        output_row["needs_label_review"] = (
            "YES"
            if row["status"] == "CAN_XAC_MINH_NHAN"
            else "NO"
        )

        writer.writerow(output_row)

# 5. Thong ke
counts = Counter()
class_counts = defaultdict(Counter)

for row in rows:
    split = row["split"]
    label = row["class"]

    counts[split] += 1
    class_counts[label][split] += 1

with open(
    OUT / "phan_bo_97_lop.csv",
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "class", "train", "val", "test",
        "exclude_review"
    ])

    for label in sorted(class_counts):
        c = class_counts[label]
        writer.writerow([
            label, c["train"], c["val"],
            c["test"], c["exclude_review"]
        ])

with open(
    OUT / "anh_can_xem_lai.csv",
    "w", encoding="utf-8-sig", newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow(["path", "class", "reason"])

    for row in quarantined:
        writer.writerow([
            row["path"],
            row["class"],
            row["near_review"]
        ])

report = [
    "BAO CAO CHIA DATASET 97 LOP - DU KIEN",
    "=" * 50,
    f"Tong anh: {len(rows)}",
    f"So lop: {len(class_counts)}",
    f"Train: {counts['train']}",
    f"Validation: {counts['val']}",
    f"Test: {counts['test']}",
    f"Anh cach ly: {counts['exclude_review']}",
    f"So nhom ro ri theo ID: {len(leakage)}",
    "",
    "LUU Y:",
    "Day la split DU KIEN, chua phai FINAL.",
    "Nhom gan trung chua duoc xac minh day du.",
    "Anh co needs_label_review=YES can xac minh nhan.",
    "Chua dam bao khong ro ri anh xoay/lat.",
    "Khong sua, xoa hay di chuyen anh goc."
]

result = "\n".join(report)

REPORT.write_text(result, encoding="utf-8")

print("\n" + result)
print("\nDa luu:", OUT)
