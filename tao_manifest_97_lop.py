
from pathlib import Path, PureWindowsPath
from collections import Counter, defaultdict
import csv

# ==================================
# 1. DUONG DAN
# ==================================

ROOT = Path(r"D:\DoanHMvUD")
DATA = ROOT / "PlantVillage"
REVIEW = ROOT / "dataset_review"

MAPPING = REVIEW / "anh_xa_thu_muc.csv"

PAPAYA = (
    REVIEW / "papaya_clean"
    / "tat_ca_anh_phan_loai.csv"
)

TEA = (
    REVIEW / "tea_grouped"
    / "tea_grouped_split_CHUAN.csv"
)

OUT = REVIEW / "dataset_97_lop"
OUT.mkdir(parents=True, exist_ok=True)

MANIFEST = OUT / "dataset_manifest_SO_BO.csv"
SUMMARY = OUT / "tong_hop_97_lop.csv"
REPORT = OUT / "bao_cao_tong_hop.txt"

EXT = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

for source in [MAPPING, PAPAYA, TEA]:
    if not source.is_file():
        raise FileNotFoundError(source)

# ==================================
# 2. BIEN THONG KE
# ==================================

class_counts = Counter()
status_counts = Counter()
split_counts = Counter()
class_status = defaultdict(Counter)
warnings = []

expected_classes = set()

total = 0
skipped = Counter()


# ==================================
# 3. HAM GHI DU LIEU
# ==================================

with open(
    MANIFEST, "w",
    newline="", encoding="utf-8-sig"
) as output:

    writer = csv.writer(output)

    writer.writerow([
        "path",
        "class",
        "split",
        "status",
        "source",
        "group_id"
    ])

    def add_image(
        path, label, split,
        status, source, group_id=""
    ):
        global total

        writer.writerow([
            str(path),
            label,
            split,
            status,
            source,
            group_id
        ])

        total += 1
        class_counts[label] += 1
        status_counts[status] += 1
        split_counts[split] += 1
        class_status[label][status] += 1

    # ==================================
    # 4. CAC DATASET THONG THUONG
    # ==================================

    print("1. Dang tong hop cac dataset...")

    with open(
        MAPPING, encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            label = row["class"]
            decision = row["decision"]

            if label:
                expected_classes.add(label)

            if not label or decision == "TAM_BO_QUA":
                skipped["Thu muc tam bo qua"] += int(
                    row["image_count"]
                )
                continue

            if decision not in {
                "DE_XUAT_DUNG", "CAN_XAC_MINH"
            }:
                continue

            relative = PureWindowsPath(
                row["folder"]
            )

            # Tea va Papaya co manifest rieng
            dataset_name = relative.parts[0]

            if dataset_name == "Tea Leaf Dataset":
                continue

            if dataset_name == "Papaya Leaf Disease Image Dataset":
                continue

            folder = DATA.joinpath(
                *relative.parts
            )

            if not folder.is_dir():
                warnings.append(
                    f"Thieu thu muc: {folder}"
                )
                continue

            # Durian da co chia train/val/test
            if dataset_name.startswith(
                "Durian_Leaf_Diseases"
            ):
                split = row["split"]
                if split not in {
                    "train", "val", "test"
                }:
                    warnings.append(
                        f"Split Durian khong hop le: {folder}"
                    )
                    continue
            else:
                # Chua chia de tranh ro ri
                split = "pending"

            if decision == "CAN_XAC_MINH":
                status = "CAN_XAC_MINH_NHAN"
            else:
                status = "CAN_KIEM_TRA_TRUNG"

            images = [
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in EXT
            ]

            expected_count = int(
                row["image_count"]
            )

            if len(images) != expected_count:
                warnings.append(
                    f"Sai so luong: {folder} "
                    f"{len(images)}/{expected_count}"
                )

            for path in sorted(images):
                add_image(
                    path, label, split,
                    status, "GENERAL"
                )

    # ==================================
    # 5. DATASET PAPAYA DA LOC
    # ==================================

    print("2. Dang nhap du lieu Papaya...")

    with open(
        PAPAYA, encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row["status"] != "DE_XUAT_SAU_KIEM_TRA":
                skipped["Papaya cach ly"] += 1
                continue

            label = "Papaya___" + {
                "Anthracnose": "Anthracnose",
                "BacterialSpot": "Bacterial_spot",
                "Curl": "Curl",
                "Healthy": "healthy",
                "Mealybug": "Mealybug",
                "Mite disease": "Mite_damage",
                "Mosaic": "Mosaic",
                "Ringspot": "Ringspot"
            }[row["class"]]

            add_image(
                row["path"],
                label,
                "pending",
                "CAN_XAC_MINH_NHAN",
                "PAPAYA_FILTERED"
            )

    # ==================================
    # 6. DATASET TEA DA GOM NHOM
    # ==================================

    print("3. Dang nhap du lieu Tea...")

    with open(
        TEA, encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if row["split"] == "exclude_eval_variant":
                skipped["Tea bien the Val/Test"] += 1
                continue

            if row["split"] not in {
                "train", "val", "test"
            }:
                warnings.append(
                    f"Split Tea khong hop le: {row['path']}"
                )
                continue

            add_image(
                row["path"],
                row["class"],
                row["split"],
                "DA_GOM_NHOM_CAN_KIEM_TRA",
                "TEA_GROUPED",
                row["group_id"]
            )

# ==================================
# 7. KIEM TRA DU 97 LOP
# ==================================

missing = sorted(
    expected_classes - set(class_counts)
)

with open(
    SUMMARY, "w",
    newline="", encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "class", "images",
        "pending_review",
        "grouped_tea",
        "general_candidates"
    ])

    for label in sorted(class_counts):

        c = class_status[label]

        writer.writerow([
            label,
            class_counts[label],
            c["CAN_XAC_MINH_NHAN"],
            c["DA_GOM_NHOM_CAN_KIEM_TRA"],
            c["CAN_KIEM_TRA_TRUNG"]
        ])

report = [
    "BAO CAO DATASET TONG HOP 97 LOP",
    "=" * 50,
    f"So lop mong doi: {len(expected_classes)}",
    f"So lop tim thay: {len(class_counts)}",
    f"So anh ung vien: {total}",
    "",
    "TRANG THAI ANH:"
]

for key, count in sorted(status_counts.items()):
    report.append(f"{key}: {count}")

report.extend([
    "",
    "SPLIT HIEN TAI:"
])

for key, count in sorted(split_counts.items()):
    report.append(f"{key}: {count}")

report.extend([
    "",
    "ANH CHUA DUOC CHON:"
])

for key, count in sorted(skipped.items()):
    report.append(f"{key}: {count}")

report.extend([
    "",
    f"So lop bi thieu: {len(missing)}",
    *missing,
    "",
    f"So canh bao: {len(warnings)}",
    *warnings[:20],
    "",
    "CHU Y:",
    "Manifest nay CHUA SAN SANG de train.",
    "pending chua duoc chia train/val/test.",
    "Chua kiem tra trung lap tren toan bo 18 cay.",
    "Papaya van can xac minh nhan.",
    "Tea can kiem tra them quan he anh nguon.",
    "Khong xoa hoac thay doi du lieu goc."
])

text = "\n".join(report)

REPORT.write_text(
    text, encoding="utf-8"
)

print("\n" + text)
print("\nDa luu ket qua tai:", OUT)
