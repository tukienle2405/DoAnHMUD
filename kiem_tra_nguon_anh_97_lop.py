
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageOps, ImageDraw
import csv
import re

ROOT = Path(r"D:\DoanHMvUD")

BASE = ROOT / "dataset_review" / "dataset_97_lop"
NEAR = BASE / "kiem_tra_gan_trung"

MANIFEST = NEAR / "dataset_manifest_NHOM_SO_BO.csv"
CROSS = NEAR / "nghi_gan_trung_khac_nhan.csv"

OUT = BASE / "kiem_tra_nguon_anh"
OUT.mkdir(parents=True, exist_ok=True)

for file in (MANIFEST, CROSS):
    if not file.is_file():
        raise FileNotFoundError(file)


def crop_name(label):
    if label.startswith("Tomato"):
        return "Tomato"
    if label.startswith("Pepper__bell"):
        return "Pepper"
    return label.split("___")[0]


def save_csv(name, columns, rows):
    with open(
        OUT / name, "w",
        newline="", encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)


def make_sheet(output, entries, title):
    # entries: danh sach (path, nhan)
    count = min(len(entries), 9)
    if count == 0:
        return

    cols = 3
    rows = (count + cols - 1) // cols

    sheet = Image.new(
        "RGB", (780, rows * 240 + 35), "white"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 8), title[:105], fill="black")

    for i, (filename, label) in enumerate(entries[:9]):
        x = (i % 3) * 260
        y = (i // 3) * 240 + 35

        draw.text(
            (x + 6, y + 3),
            label[:34],
            fill="black"
        )

        try:
            with Image.open(filename) as image:
                image = ImageOps.exif_transpose(image)
                image = ImageOps.contain(
                    image.convert("RGB"),
                    (240, 170)
                )
                sheet.paste(image, (x + 8, y + 25))

            draw.text(
                (x + 6, y + 203),
                Path(filename).name[:35],
                fill="black"
            )

        except Exception as exc:
            draw.text(
                (x + 8, y + 80),
                str(exc)[:28],
                fill="red"
            )

    sheet.save(output, quality=90)


# =====================================
# 1. KIEM TRA QUY LUAT TEN FILE
# =====================================

families = defaultdict(list)
pending_counts = Counter()
pending_total = 0

with open(
    MANIFEST, encoding="utf-8-sig", newline=""
) as f:
    for row in csv.DictReader(f):
        if row["split"] != "pending":
            continue

        pending_total += 1
        path = Path(row["path"])
        label = row["class"]
        crop = crop_name(label)

        pending_counts[crop] += 1

        # Chi kiem tra mau ten dang 123_456.jpg
        match = re.fullmatch(
            r"(\d+)_(\d+)", path.stem
        )

        if match is None:
            continue

        prefix = match.group(1)
        suffix = int(match.group(2))

        key = (
            crop, label,
            str(path.parent), prefix
        )

        families[key].append({
            "number": suffix,
            "path": str(path),
            "near_group": row["near_group_id"]
        })

print("Anh pending:", pending_total)
print("Nhom ten file tim thay:", len(families))

family_rows = []
large_families = defaultdict(list)

for key, items in families.items():
    crop, label, folder, prefix = key

    if len(items) < 2:
        continue

    near_ids = {
        x["near_group"]
        for x in items
    }

    family_rows.append([
        crop, label, folder, prefix,
        len(items), len(near_ids),
        items[0]["path"]
    ])

    if len(items) >= 3:
        large_families[crop].append(
            (key, items)
        )

family_rows.sort(
    key=lambda x: (-x[4], x[0])
)

save_csv(
    "nhom_theo_ten_file.csv",
    [
        "crop", "class", "folder", "prefix",
        "image_count", "near_group_count",
        "example_path"
    ],
    family_rows
)

# Tao bang anh cua toi da 2 nhom moi cay
sheet_count = 0

for crop, candidates in sorted(
    large_families.items()
):
    candidates.sort(
        key=lambda x: -len(x[1])
    )

    for j, (key, items) in enumerate(
        candidates[:2], 1
    ):
        items = sorted(
            items, key=lambda x: x["number"]
        )

        n = min(len(items), 9)

        if n == 1:
            chosen = [items[0]]
        else:
            chosen = [
                items[
                    round(
                        i * (len(items) - 1)
                        / (n - 1)
                    )
                ]
                for i in range(n)
            ]

        entries = [
            (item["path"], key[1])
            for item in chosen
        ]

        filename = (
            f"ten_file_{crop}_{j}.jpg"
        )

        make_sheet(
            OUT / filename,
            entries,
            f"{crop} | prefix={key[3]} "
            f"| total={len(items)}"
        )

        sheet_count += 1


# =====================================
# 2. KIEM TRA CAC ANH KHAC NHAN
# =====================================

cross_groups = defaultdict(list)

with open(
    CROSS, encoding="utf-8-sig", newline=""
) as f:
    for row in csv.DictReader(f):
        key = (
            row["phash"],
            row["dhash"]
        )
        cross_groups[key].append(row)

cross_report = []
cross_images = 0

for index, (key, items) in enumerate(
    cross_groups.items(), 1
):
    labels = sorted({
        x["class"] for x in items
    })

    if len(labels) < 2:
        continue

    cross_report.append([
        index,
        key[0], key[1],
        len(items),
        " | ".join(labels)
    ])

    entries = [
        (x["path"], x["class"])
        for x in items
    ]

    make_sheet(
        OUT / f"khac_nhan_{index:02d}.jpg",
        entries,
        f"Nghi van khac nhan - nhom {index}"
    )

    cross_images += 1

save_csv(
    "nhom_nghi_khac_nhan.csv",
    [
        "group", "phash", "dhash",
        "image_count", "classes"
    ],
    cross_report
)


# =====================================
# 3. BAO CAO
# =====================================

report = [
    "BAO CAO KIEM TRA NGUON ANH 97 LOP",
    "=" * 48,
    f"Tong anh pending: {pending_total}",
    f"Nhom ten dang so_so: {len(families)}",
    f"Nhom ten co >= 2 anh: {len(family_rows)}",
    f"Bang anh theo ten da tao: {sheet_count}",
    f"Nhom nghi khac nhan: {len(cross_report)}",
    f"Bang anh khac nhan: {cross_images}",
    "",
    "SO NHOM TEN FILE THEO CAY:"
]

for crop in sorted(pending_counts):
    groups = sum(
        1 for row in family_rows
        if row[0] == crop
    )

    report.append(
        f"{crop}: {pending_counts[crop]} anh pending, "
        f"{groups} nhom ten >=2 anh"
    )

report.extend([
    "",
    "NHOM TEN FILE LON NHAT:"
])

for row in family_rows[:15]:
    report.append(
        f"{row[0]} / {row[1]} / "
        f"prefix={row[3]} / "
        f"images={row[4]} / "
        f"near_groups={row[5]}"
    )

report.extend([
    "",
    "CHU Y:",
    "Trung tien to ten file khong chung minh "
    "cung mot anh nguon.",
    "Can xem bang anh truoc khi gom nhom.",
    "Chua thay doi train/val/test.",
    "Khong xoa hoac di chuyen anh goc."
])

result = "\n".join(report)

(OUT / "bao_cao_nguon_anh.txt").write_text(
    result, encoding="utf-8"
)

print("\n" + result)
print("\nDa luu tai:", OUT)
