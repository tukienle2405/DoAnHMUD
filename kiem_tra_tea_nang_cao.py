
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import cv2
import csv

ROOT = Path(r"D:\DoanHMvUD")
FOLDER = ROOT / "dataset_review" / "tea_review"
INPUT = FOLDER / "tea_nghi_giong_dhash.csv"
OUT = FOLDER / "kiem_tra_nang_cao"
OUT.mkdir(parents=True, exist_ok=True)


def phash(path):
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize(
            (32, 32), Image.Resampling.LANCZOS
        )
        arr = np.asarray(img, dtype=np.float32)

    coefficients = cv2.dct(arr)[:8, :8]
    values = coefficients.flatten()
    median = np.median(values[1:])

    result = 0

    for value in values:
        result = (result << 1) | int(value > median)

    return result


def save_csv(name, header, rows):
    with open(
        OUT / name, "w",
        newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def draw_image(sheet, path, x, y, label):
    draw = ImageDraw.Draw(sheet)
    draw.text((x, y), label[:40], fill="black")

    try:
        with Image.open(path) as img:
            img = ImageOps.contain(
                img.convert("RGB"), (360, 190)
            )
            sheet.paste(img, (x, y + 20))

        draw.text(
            (x, y + 215),
            Path(path).name[:45],
            fill="black"
        )

    except Exception as exc:
        draw.text(
            (x, y + 50),
            str(exc)[:40],
            fill="red"
        )


if not INPUT.exists():
    raise FileNotFoundError(INPUT)

groups = defaultdict(list)

with open(INPUT, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        groups[row["dhash"]].append(row)

print("So nhom dHash:", len(groups))

cache = {}
errors = []
pairs = []

for number, (dhash, items) in enumerate(
    groups.items(), 1
):
    for item in items:
        path = item["path"]

        if path not in cache:
            try:
                cache[path] = phash(path)
            except Exception as exc:
                errors.append([path, str(exc)])

    # So sanh nhung anh khac lop
    for i in range(len(items)):
        a = items[i]

        for b in items[i + 1:]:
            if a["class"] == b["class"]:
                continue

            p1, p2 = a["path"], b["path"]

            if p1 not in cache or p2 not in cache:
                continue

            distance = (
                cache[p1] ^ cache[p2]
            ).bit_count()

            if distance <= 5:
                priority = "CAO"
            elif distance <= 10:
                priority = "TRUNG_BINH"
            else:
                priority = "THAP"

            pairs.append({
                "dhash": dhash,
                "distance": distance,
                "priority": priority,
                "class1": a["class"],
                "class2": b["class"],
                "path1": p1,
                "path2": p2
            })

    if number % 500 == 0:
        print("Da kiem tra:", number, "nhom")


pairs.sort(key=lambda x: x["distance"])

save_csv(
    "tea_phash_uu_tien.csv",
    [
        "dhash", "phash_distance", "priority",
        "class1", "class2", "path1", "path2"
    ],
    [
        [
            p["dhash"], p["distance"], p["priority"],
            p["class1"], p["class2"],
            p["path1"], p["path2"]
        ]
        for p in pairs
    ]
)

save_csv(
    "anh_loi.csv",
    ["path", "error"],
    errors
)

# Lay mot cap dai dien cho moi nhom dHash
representatives = []
used_groups = set()

for p in pairs:
    if p["dhash"] in used_groups:
        continue

    used_groups.add(p["dhash"])
    representatives.append(p)

    if len(representatives) == 12:
        break

# Tao 3 bang anh, moi bang 4 cap
for page in range(3):
    selected = representatives[page * 4:(page + 1) * 4]

    if not selected:
        break

    sheet = Image.new(
        "RGB", (800, 4 * 270), "white"
    )
    draw = ImageDraw.Draw(sheet)

    for index, p in enumerate(selected):
        y = index * 270

        draw.text(
            (10, y + 2),
            f"pHash distance = {p['distance']}",
            fill="black"
        )

        draw_image(
            sheet, p["path1"], 10, y + 25,
            p["class1"]
        )

        draw_image(
            sheet, p["path2"], 405, y + 25,
            p["class2"]
        )

    sheet.save(
        OUT / f"so_sanh_tea_{page + 1}.jpg",
        quality=90
    )

counts = Counter(
    p["priority"] for p in pairs
)

report = [
    "BAO CAO KIEM TRA TEA NANG CAO",
    "=" * 45,
    f"So nhom dHash dau vao: {len(groups)}",
    f"So anh duoc tinh pHash: {len(cache)}",
    f"So cap so sanh khac lop: {len(pairs)}",
    f"Cap uu tien CAO (pHash <= 5): {counts['CAO']}",
    f"Cap TRUNG BINH (6-10): {counts['TRUNG_BINH']}",
    f"Cap THAP (>10): {counts['THAP']}",
    f"So anh loi: {len(errors)}",
    "",
    "pHash chi phuc vu sang loc.",
    "Khong ket luan trung anh hoac sai nhan tu pHash.",
    "Chi kiem tra anh mau trong bao cao dHash.",
    "Khong sua, xoa hoac di chuyen anh goc."
]

report_text = "\n".join(report)

(OUT / "bao_cao_nang_cao.txt").write_text(
    report_text, encoding="utf-8"
)

print("\n" + report_text)
print("\nDa luu ket qua tai:", OUT)
