
from pathlib import Path, PureWindowsPath
from PIL import Image, ImageOps, ImageDraw
import csv
import random

ROOT = Path(r"D:\DoanHMvUD")
DATA = ROOT / "PlantVillage"
MAPPING = ROOT / "dataset_review" / "anh_xa_thu_muc.csv"
OUTPUT = ROOT / "dataset_review" / "mau_can_xac_minh"

OUTPUT.mkdir(parents=True, exist_ok=True)

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
random_generator = random.Random(42)

# Gom duong dan theo lop can xac minh
classes = {}

with open(MAPPING, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["decision"] != "CAN_XAC_MINH":
            continue

        class_name = row["class"]
        folder = DATA.joinpath(
            *PureWindowsPath(row["folder"]).parts
        )

        classes.setdefault(class_name, []).append(folder)

print(f"Tim thay {len(classes)} lop can kiem tra")

for class_name, folders in sorted(classes.items()):
    images = []

    for folder in folders:
        if not folder.is_dir():
            print(f"Khong tim thay: {folder}")
            continue

        images.extend(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXT
        )

    if not images:
        print(f"Khong co anh: {class_name}")
        continue

    # Uu tien mot vai ten file dang nghi neu co
    suspicious = [
        p for p in images
        if "increased" in p.name.lower()
        and "Anthracnose" not in class_name
    ]

    selected = suspicious[:3]

    remaining = [
        p for p in images if p not in selected
    ]

    selected += random_generator.sample(
        remaining,
        min(12 - len(selected), len(remaining))
    )

    # Moi anh mau co 4 cot x 3 hang
    width, height = 220, 245
    sheet = Image.new(
        "RGB", (width * 4, height * 3), "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, path in enumerate(selected):
        x = (i % 4) * width
        y = (i // 4) * height

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = ImageOps.contain(img, (210, 205))

                px = x + (width - img.width) // 2
                py = y + 5
                sheet.paste(img, (px, py))

            filename = path.name[:28]
            draw.text(
                (x + 5, y + 211),
                filename,
                fill="black"
            )

        except Exception as exc:
            print("Anh loi:", path, exc)

    save_path = OUTPUT / f"{class_name}.jpg"
    sheet.save(save_path, quality=90)

    print(
        f"{class_name}: "
        f"{len(images)} anh, da tao anh mau"
    )

print("\nHOAN THANH!")
print("Ket qua:", OUTPUT)
