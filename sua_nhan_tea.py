
from pathlib import Path
from collections import defaultdict, Counter
import csv

ROOT = Path(r"D:\DoanHMvUD")
FOLDER = ROOT / "dataset_review" / "tea_grouped"

INPUT = FOLDER / "tea_grouped_split_DU_KIEN.csv"
OUTPUT = FOLDER / "tea_grouped_split_CHUAN.csv"

LABELS = {
    "Tea___Blister_Blight": "Tea___Blister_blight",
    "Tea___Brown_Blight": "Tea___Brown_blight",
    "Tea___Leaf_Red_Rust": "Tea___Red_rust",
    "Tea___Red_Spider_Mite": "Tea___Red_spider_mite",
    "Tea___Tea_Mosquito_Bug": "Tea___Tea_mosquito_bug",
    "Tea___healthy": "Tea___healthy",
}

groups = defaultdict(set)
counts = Counter()
total = 0

with open(INPUT, encoding="utf-8-sig", newline="") as source:
    reader = csv.DictReader(source)

    with open(
        OUTPUT, "w", encoding="utf-8-sig", newline=""
    ) as target:

        writer = csv.DictWriter(
            target,
            fieldnames=reader.fieldnames
        )
        writer.writeheader()

        for row in reader:
            old_label = row["class"]

            if old_label not in LABELS:
                raise ValueError(
                    f"Nhan khong xac dinh: {old_label}"
                )

            row["class"] = LABELS[old_label]

            label = row["class"]
            group_id = row["group_id"]
            split = row["split"]

            if split not in {
                "train", "val", "test",
                "exclude_eval_variant"
            }:
                raise ValueError(
                    f"Split khong hop le: {split}"
                )

            # Group ID chi duy nhat trong tung lop
            key = (label, group_id)

            if split in {"train", "val", "test"}:
                groups[key].add(split)

            counts[split] += 1
            writer.writerow(row)
            total += 1

leakage = [
    key for key, splits in groups.items()
    if len(splits) > 1
]

print("\n===== KIEM TRA DATASET CHE =====")
print("Tong so file:", total)
print("So nhom:", len(groups))

for split in [
    "train", "val", "test",
    "exclude_eval_variant"
]:
    print(f"{split}: {counts[split]}")

print("So nhom bi chia qua nhieu tap:", len(leakage))

if total != 80329:
    print("CANH BAO: Tong anh khac 80329!")

if len(groups) != 9000:
    print("CANH BAO: So nhom khac 9000!")

if leakage:
    print("CANH BAO: Phat hien nguy co ro ri!")
    for item in leakage[:10]:
        print(item)
else:
    print("Khong phat hien ro ri theo group_id.")

print("\nDa luu:", OUTPUT)
print("Du lieu goc khong bi thay doi.")
    