
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(r"D:\DoanHMvUD")

DATA = (
    ROOT / "PlantVillage"
    / "Tea Leaf Dataset"
    / "Tea Leaf Dataset"
)

OUT = (
    ROOT / "dataset_review"
    / "tea_review"
    / "kiem_tra_nguon"
)
OUT.mkdir(parents=True, exist_ok=True)

classes = {
    "Blister_Blight": (
        DATA / "Diseased Leaves" / "Blister_Blight",
        "Blister_Blight"
    ),
    "Red_Spider_Mite": (
        DATA / "Diseased Leaves" / "Red_Spider_Mite",
        "Red_Spider_Mite"
    ),
    "Healthy": (
        DATA / "Healthy Leaves" / "Healthy_leaves",
        "Healthy_leaf"
    ),
}

# Vi du voi anh so 137
base_index = 137

patterns = {
    # Gia thuyet 1: cac bien the cach nhau 1500 anh
    "cach_1500": [
        base_index + k * 1500 for k in range(9)
    ],

    # Gia thuyet 2: 9 bien the duoc dat lien tiep
    "lien_tiep_9": [
        (base_index - 1) * 9 + k + 1
        for k in range(9)
    ]
}

for class_name, (folder, prefix) in classes.items():

    for pattern_name, numbers in patterns.items():

        sheet = Image.new(
            "RGB", (750, 780), "white"
        )
        draw = ImageDraw.Draw(sheet)

        for i, number in enumerate(numbers):
            filename = f"{prefix}{number}.jpg"
            path = folder / filename

            x = (i % 3) * 250
            y = (i // 3) * 260

            draw.text(
                (x + 8, y + 5),
                filename,
                fill="black"
            )

            if path.exists():
                with Image.open(path) as img:
                    img = ImageOps.contain(
                        img.convert("RGB"),
                        (230, 220)
                    )
                    px = x + (250 - img.width) // 2
                    sheet.paste(img, (px, y + 28))
            else:
                draw.text(
                    (x + 8, y + 65),
                    "Khong tim thay anh",
                    fill="red"
                )

        output = OUT / (
            f"{class_name}_{pattern_name}.jpg"
        )
        sheet.save(output, quality=90)
        print("Da tao:", output.name)

print("\nHOAN THANH")
print("Thu muc:", OUT)
print("Chi kiem tra cach danh so, khong thay doi anh goc.")
