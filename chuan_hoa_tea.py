
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageOps, ImageDraw
import csv
import random
import re

ROOT = Path(r"D:\DoanHMvUD")
DATA = ROOT / "PlantVillage" / "Tea Leaf Dataset" / "Tea Leaf Dataset"
OUT = ROOT / "dataset_review" / "tea_grouped"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = {
    "Blister_Blight": DATA / "Diseased Leaves" / "Blister_Blight",
    "Brown_Blight": DATA / "Diseased Leaves" / "Brown_Blight",
    "Leaf_Red_Rust": DATA / "Diseased Leaves" / "Leaf_Red_Rust",
    "Red_Spider_Mite": DATA / "Diseased Leaves" / "Red_Spider_Mite",
    "Tea_Mosquito_Bug": DATA / "Diseased Leaves" / "Tea_Mosquito_Bug",
    "Healthy": DATA / "Healthy Leaves" / "Healthy_leaves",
}

GROUP_SIZE = 1500
SEED = 42
EXT = {".jpg", ".jpeg", ".png", ".webp"}

def get_number(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None

def create_sheet(cls, indexed):
    # Kiem tra cac anh cach nhau 1500
    numbers = [137 + k * GROUP_SIZE for k in range(9)]
    sheet = Image.new("RGB", (750, 780), "white")
    draw = ImageDraw.Draw(sheet)

    for i, number in enumerate(numbers):
        x = (i % 3) * 250
        y = (i // 3) * 260
        path = indexed.get(number)

        draw.text((x + 6, y + 5), f"{cls} - {number}", fill="black")

        if path is None:
            draw.text((x + 8, y + 70), "Khong co anh", fill="red")
            continue

        try:
            with Image.open(path) as img:
                img = ImageOps.contain(
                    img.convert("RGB"), (230, 220)
                )
                px = x + (250 - img.width) // 2
                sheet.paste(img, (px, y + 30))
        except Exception as exc:
            draw.text((x + 8, y + 70), str(exc)[:25], fill="red")

    sheet.save(OUT / f"kiem_tra_{cls}.jpg", quality=90)

def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

manifest = []
report = []
errors = []

for cls, folder in CLASSES.items():
    if not folder.is_dir():
        raise FileNotFoundError(folder)

    indexed = {}

    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in EXT:
            continue

        number = get_number(path)

        if number is None or not 1 <= number <= 13500:
            errors.append([cls, str(path), "So thu tu khong hop le"])
            continue

        if number in indexed:
            errors.append([cls, str(path), "Trung so thu tu"])
            continue

        indexed[number] = path

    # Tao anh kiem tra cho ca 6 lop
    create_sheet(cls, indexed)

    # Gom anh theo ID anh nguon
    groups = defaultdict(list)

    for number, path in indexed.items():
        group_id = (number - 1) % GROUP_SIZE + 1
        variant = (number - 1) // GROUP_SIZE
        groups[group_id].append((variant, path))

    ids = sorted(groups)
    rng = random.Random(SEED)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])

    counts = defaultdict(int)

    for group_id, items in groups.items():
        items.sort(key=lambda x: x[0])

        # Chon mot anh dai dien cho Val/Test
        representative = items[0][1]

        for variant, path in items:
            if group_id in train_ids:
                split = "train"
            elif group_id in val_ids:
                split = (
                    "val" if path == representative
                    else "exclude_eval_variant"
                )
            else:
                split = (
                    "test" if path == representative
                    else "exclude_eval_variant"
                )

            counts[split] += 1

            manifest.append([
                str(path),
                f"Tea___{cls if cls != 'Healthy' else 'healthy'}",
                group_id,
                variant,
                split
            ])

    report.append(
        f"{cls}: {len(indexed)} anh, {len(groups)} nhom, "
        f"train={counts['train']}, "
        f"val={counts['val']}, test={counts['test']}"
    )

save_csv(
    OUT / "tea_grouped_split_DU_KIEN.csv",
    ["path", "class", "group_id", "variant", "split"],
    manifest
)

save_csv(
    OUT / "loi_danh_so.csv",
    ["class", "path", "error"],
    errors
)

text = "\n".join([
    "BAO CAO GOM NHOM ANH CHE",
    "=" * 45,
    *report,
    "",
    f"So file co van de danh so: {len(errors)}",
    "",
    "Can xac minh quy luat 1500 o ca 6 lop.",
    "Chua xac nhan anh dai dien la anh goc.",
    "Manifest chi la phuong an chia du kien.",
    "Khong xoa hoac di chuyen anh."
])

(OUT / "bao_cao_gom_nhom.txt").write_text(
    text, encoding="utf-8"
)

print(text)
print("\nKet qua:", OUT)
