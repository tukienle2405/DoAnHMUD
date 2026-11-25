
from pathlib import Path
from collections import defaultdict
import csv
import os

ROOT = Path(r"D:\DoanHMvUD\PlantVillage")
OUTPUT = ROOT.parent / "dataset_inventory.csv"

EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}

if not ROOT.is_dir():
    raise FileNotFoundError(
        f"Khong tim thay: {ROOT}"
    )

rows = []
group_totals = defaultdict(int)
total_images = 0

for folder, subdirs, filenames in os.walk(ROOT):
    images = sorted(
        name for name in filenames
        if Path(name).suffix.lower() in EXTENSIONS
    )

    if not images:
        continue

    relative = Path(folder).relative_to(ROOT)
    dataset_name = relative.parts[0]
    count = len(images)

    group_totals[dataset_name] += count
    total_images += count

    rows.append([
        dataset_name,
        str(relative),
        count,
        images[0]
    ])

rows.sort(key=lambda row: (row[0], row[1]))

with open(
    OUTPUT, "w", newline="", encoding="utf-8-sig"
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "dataset",
        "class_folder",
        "image_count",
        "example_filename"
    ])
    writer.writerows(rows)

print("\n===== THONG KE DATASET =====")
for name, count in sorted(group_totals.items()):
    print(f"{name}: {count:,} anh")

print(f"\nTong so file anh: {total_images:,}")
print(f"So thu muc chua anh: {len(rows)}")
print(f"\nDa luu bao cao: {OUTPUT}")
print("Luu y: Chua kiem tra anh hong/trung lap.")
