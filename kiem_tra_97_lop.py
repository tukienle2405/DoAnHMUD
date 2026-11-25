
from pathlib import Path, PureWindowsPath
from collections import defaultdict, Counter
from PIL import Image
import argparse
import csv
import random

ROOT = Path(r"D:\DoanHMvUD")
DATA = ROOT / "PlantVillage"
INPUT = ROOT / "dataset_inventory.csv"
OUTPUT = ROOT / "dataset_review"
OUTPUT.mkdir(exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".bmp",
        ".webp", ".tif", ".tiff"}

# Nhan dien cac dataset theo ten thu muc goc
GROUPS = {
    "A Machine  Learning Dataset for Classification of Common Coffee Leaf Diseases in Uganda": "Coffee",
    "Banana Leaf Image Dataset for Classification of Si": "Banana",
    "Comprehensive Lemon Leaf Disease Dataset for Advanced Detection and Sustainable Agriculture": "Lemon",
    "Durian_Leaf_Diseases_Release_20261017_v2": "Durian",
    "Image Dataset for Detection and classification of Diseases of Guava Fruits and Leaves": "Guava",
    "Papaya Leaf Disease Image Dataset": "Papaya",
    "Rubber_Leaf_Dataset": "Rubber",
    "Sugarcane Leaf Image Dataset": "Sugarcane",
    "Tea Leaf Dataset": "Tea",
    "cashew and jackfruit": "Mixed",
    "chileleaf": "Chili",
    "corndisease": "Corn",
    "mago": "Mango",
    "pepper": "Pepper",
    "potato": "Potato",
    "rice_leaf_diseases": "Rice",
    "tomato": "Tomato",
}

# Anh xa ten thu muc sang nhan chuan
LABELS = {
    "Coffee": {
        "Health leaves": "healthy",
        "leaf rust": "Rust",
        "phoma": "Phoma"
    },
    "Banana": {
        "cordana": "Cordana",
        "sanas": "healthy",
        "sigatoka": "Sigatoka"
    },
    "Lemon": {
        "Anthracnose": "Anthracnose",
        "Bacterial Blight": "Bacterial_blight",
        "Citrus Canker": "Citrus_canker",
        "Curl Virus": "Curl_virus",
        "Deficiency Leaf": "Deficiency_leaf",
        "Dry Leaf": "Dry_leaf",
        "Healthy Leaf": "healthy",
        "Sooty Mould": "Sooty_mould",
        "Spider Mites": "Spider_mites"
    },
    "Durian": {
        "Leaf_Algal": "Algal",
        "Leaf_Blight": "Blight",
        "Leaf_Colletotrichum": "Colletotrichum",
        "Leaf_Healthy": "healthy",
        "Leaf_Phomopsis": "Phomopsis",
        "Leaf_Rhizoctonia": "Rhizoctonia"
    },
    "Guava": {
        "Anthracnose": "Anthracnose",
        "Canker": "Canker",
        "Dot": "Dot",
        "Rust": "Rust",
        "Healthy": "healthy"
    },
    "Papaya": {
        "Anthracnose": "Anthracnose",
        "BacterialSpot": "Bacterial_spot",
        "Curl": "Curl",
        "Healthy": "healthy",
        "Mealybug": "Mealybug",
        "Mite disease": "Mite_damage",
        "Mosaic": "Mosaic",
        "Ringspot": "Ringspot"
    },
    "Sugarcane": {
        "Banded Chlorosis": "Banded_chlorosis",
        "Brown Spot": "Brown_spot",
        "BrownRust": "Brown_rust",
        "Grassy shoot": "Grassy_shoot",
        "Pokkah Boeng": "Pokkah_boeng",
        "Sett Rot": "Sett_rot",
        "Viral Disease": "Viral_disease",
        "Yellow Leaf": "Yellow_leaf",
        "smut": "Smut",
        "Dried Leaves": "Dried_leaves",
        "Healthy Leaves": "healthy"
    },
    "Tea": {
        "Blister_Blight": "Blister_blight",
        "Brown_Blight": "Brown_blight",
        "Leaf_Red_Rust": "Red_rust",
        "Red_Spider_Mite": "Red_spider_mite",
        "Tea_Mosquito_Bug": "Tea_mosquito_bug",
        "Healthy_leaves": "healthy"
    },
    "Cashew": {
        "cashew fungal infection": "Fungal_infection",
        "cashew healthy": "healthy"
    },
    "Jackfruit": {
        "jackfruit algal": "Algal",
        "jackfruit anthracnose": "Anthracnose",
        "jackfruit healthy": "healthy",
        "jackfruit leaf spot": "Leaf_spot",
        "jackfruit phyllosticta": "Phyllosticta",
        "jackfruit sooty mold": "Sooty_mould"
    },
    "Chili": {
        "Bacterial_Spot": "Bacterial_spot",
        "Cercospora_Leaf_Spot": "Cercospora_leaf_spot",
        "Curl_Virus": "Curl_virus",
        "Healthy_Leaf": "healthy"
    },
    "Corn": {
        "Blight": "Blight",
        "Common_Rust": "Common_rust",
        "Gray_Leaf_Spot": "Gray_leaf_spot",
        "Healthy": "healthy"
    },
    "Mango": {
        "Anthracnose": "Anthracnose",
        "Bacterial Canker": "Bacterial_canker",
        "Cutting Weevil": "Cutting_weevil",
        "die back": "Die_back",
        "Gall Midge": "Gall_midge",
        "Healthy": "healthy",
        "Powdery Mildew": "Powdery_mildew",
        "Sooty Mould": "Sooty_mould"
    },
    "Pepper": {
        "Pepper__bell___Bacterial_spot": "Bacterial_spot",
        "Pepper__bell___healthy": "healthy"
    },
    "Potato": {
        "Potato___Early_blight": "Early_blight",
        "Potato___Late_blight": "Late_blight",
        "Potato___healthy": "healthy"
    },
    "Rice": {
        "Bacterial leaf blight": "Bacterial_leaf_blight",
        "Brown spot": "Brown_spot",
        "Leaf smut": "Leaf_smut"
    },
    "Tomato": {
        "Tomato_Bacterial_spot": "Bacterial_spot",
        "Tomato_Early_blight": "Early_blight",
        "Tomato_Late_blight": "Late_blight",
        "Tomato_Leaf_Mold": "Leaf_mold",
        "Tomato_Septoria_leaf_spot": "Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite": "Spider_mites",
        "Tomato__Target_Spot": "Target_spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus": "Yellow_leaf_curl_virus",
        "Tomato__Tomato_mosaic_virus": "Mosaic_virus",
        "Tomato_healthy": "healthy"
    },
    "Rubber": {
        "Anthracnose": "Anthracnose",
        "Dry_Leaf": "Dry_leaf",
        "Healthy": "healthy",
        "Leaf_Spot": "Leaf_spot"
    }
}


def classify(relative):
    parts = PureWindowsPath(relative).parts
    group = GROUPS.get(parts[0])

    if not group:
        return "", "TAM_BO_QUA", "Dataset chua xac dinh", ""

    if group == "Mixed":
        if parts[1] == "cashew leaf":
            group = "Cashew"
        elif parts[1] == "jackfruit leaf":
            group = "Jackfruit"

    last = parts[-1]
    mapped = LABELS.get(group, {}).get(last)

    # Giu ten lop cu dung voi checkpoint 41 lop
    if group == "Pepper" and mapped:
        label = last
    elif group == "Potato" and mapped:
        label = last
    elif group == "Tomato" and mapped:
        label = last
    elif mapped:
        label = f"{group}___{mapped}"
    else:
        return "", "TAM_BO_QUA", "Chua xac minh nhan", ""

    split = ""

    if group == "Durian":
        split = parts[1]  # train, val, test

    if group == "Lemon":
        if "Aumented Dataset" in parts:
            return label, "TAM_BO_QUA", "Anh augmented", split

    if group == "Guava":
        if not any(
            p.lower() == "guava_leaf" for p in parts
        ):
            return label, "TAM_BO_QUA", "Anh qua oi", split

        if "Augmented" in parts:
            return label, "TAM_BO_QUA", "Anh augmented", split

    if group == "Papaya":
        if "augmented" in parts:
            return label, "TAM_BO_QUA", "Anh augmented", split

    if group == "Rubber":
        if "Raw_Dataset" not in parts:
            return label, "TAM_BO_QUA", "Anh compressed", split

    if group == "Mango":
        if "Mango Dataset" not in parts:
            return label, "TAM_BO_QUA", "Anh xoai bo sung", split

    if group == "Tea":
        return label, "CAN_XAC_MINH", "Nguon anh va augmentation", split

    if group == "Banana" and last == "sanas":
        return label, "CAN_XAC_MINH", "Xac nhan sanas la healthy", split

    if group == "Guava" and last == "Dot":
        return label, "CAN_XAC_MINH", "Xac nhan nhan Dot", split

    if group == "Sugarcane" and last == "Sett Rot":
        return label, "CAN_XAC_MINH", "Kiem tra co phai anh la", split

    if group == "Papaya":
        return label, "CAN_XAC_MINH", "Kiem tra nhan anh goc", split

    return label, "DE_XUAT_DUNG", "Can kiem tra trung va ro ri", split


def save_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main(sample_per_folder):
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    with open(INPUT, encoding="utf-8-sig", newline="") as f:
        inventory = list(csv.DictReader(f))

    mapped_rows = []
    sample_errors = []
    class_counts = defaultdict(int)
    class_status = defaultdict(set)
    status_counts = Counter()
    rng = random.Random(42)

    for row in inventory:
        relative = row["class_folder"]
        count = int(row["image_count"])
        label, decision, reason, split = classify(relative)

        folder = DATA.joinpath(
            *PureWindowsPath(relative).parts
        )

        if not folder.is_dir():
            decision = "LOI_DUONG_DAN"
            reason = "Khong tim thay thu muc"

        mapped_rows.append([
            relative, label, count,
            decision, reason, split
        ])

        status_counts[decision] += count

        if decision in {"DE_XUAT_DUNG", "CAN_XAC_MINH"}:
            class_counts[label] += count
            class_status[label].add(decision)

            if not folder.is_dir():
                continue

            images = [
                p for p in folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in EXTS
            ]

            for path in rng.sample(
                images, min(sample_per_folder, len(images))
            ):
                try:
                    with Image.open(path) as img:
                        img.load()
                        if img.width < 64 or img.height < 64:
                            raise ValueError("Anh qua nho")
                except Exception as exc:
                    sample_errors.append([
                        label, str(path), str(exc)
                    ])

    save_csv(
        OUTPUT / "anh_xa_thu_muc.csv",
        ["folder", "class", "image_count",
         "decision", "reason", "split"],
        mapped_rows
    )

    class_rows = [
        [
            name, count,
            "CAN_XAC_MINH"
            if "CAN_XAC_MINH" in class_status[name]
            else "DE_XUAT_DUNG"
        ]
        for name, count in sorted(class_counts.items())
    ]

    save_csv(
        OUTPUT / "tong_hop_lop.csv",
        ["class", "images", "status"],
        class_rows
    )

    save_csv(
        OUTPUT / "anh_loi_mau.csv",
        ["class", "path", "error"],
        sample_errors
    )

    expected = sum(len(v) for v in LABELS.values())
    actual = len(class_counts)

    report = [
        "BAO CAO KIEM TRA DATASET",
        "=" * 45,
        f"So file anh thong ke: {sum(int(r['image_count']) for r in inventory):,}",
        f"So thu muc: {len(inventory)}",
        f"So lop du kien: {expected}",
        f"So lop anh xa duoc: {actual}",
        "",
        "PHAN LOAI ANH:"
    ]

    for status, count in sorted(status_counts.items()):
        report.append(f"{status}: {count:,} anh")

    report.extend([
        "",
        f"So anh loi trong mau: {len(sample_errors)}",
        "",
        "CHU Y:",
        "Chua kiem tra toan bo anh.",
        "Chua kiem tra trung lap hash/perceptual hash.",
        "Chua xac minh nhan bang hinh anh.",
        "CAN_XAC_MINH chua duoc phe duyet de train.",
        "Khong tu dong xoa, doi ten hay di chuyen anh."
    ])

    (OUTPUT / "bao_cao.txt").write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n".join(report))
    print(f"\nDa luu ket qua tai: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample", type=int, default=5,
        help="So anh kiem tra moi thu muc"
    )
    args = parser.parse_args()
    main(max(0, args.sample))
