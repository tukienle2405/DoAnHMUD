
# -*- coding: utf-8 -*-
"""
kiem_tra_dataset.py

Kiem tra 26 lop moi cua 6 cay.
Co the them 15 lop cu bang --include-old.

- Doc va giai ma tung anh
- Thu OpenCV neu Pillow khong doc duoc
- Ghi anh hop le, anh loi, anh trung
- Tao manifest duong dan cho training
- Tuy chon xuat anh JPG sach bang --export

Khong sua hay xoa anh goc.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime

from PIL import Image

ROOT = Path(r"D:\DoanHMvUD")
PLANT = ROOT / "PlantVillage"
REPORT = ROOT / "dataset_audit"

COFFEE = (
    PLANT
    / "A Machine  Learning Dataset for Classification "
      "of Common Coffee Leaf Diseases in Uganda"
    / "coffee dataset"
)

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

# Ten nhan phai khop chinh xac voi training
CLASS_DIRS = {
    # CA PHE
    "Coffee___healthy":
        COFFEE / "Health leaves",
    "Coffee___Rust":
        COFFEE / "leaf rust",
    "Coffee___Phoma":
        COFFEE / "phoma",

    # NGO
    "Corn___Blight":
        PLANT / "corndisease" / "Blight",
    "Corn___Common_rust":
        PLANT / "corndisease" / "Common_Rust",
    "Corn___Gray_leaf_spot":
        PLANT / "corndisease" / "Gray_Leaf_Spot",
    "Corn___healthy":
        PLANT / "corndisease" / "Healthy",

    # OT CAY
    "Chili___Bacterial_spot":
        PLANT / "chileleaf" / "Bacterial_Spot",
    "Chili___Cercospora_leaf_spot":
        PLANT / "chileleaf" / "Cercospora_Leaf_Spot",
    "Chili___Curl_virus":
        PLANT / "chileleaf" / "Curl_Virus",
    "Chili___healthy":
        PLANT / "chileleaf" / "Healthy_Leaf",

    # XOAI
    "Mango___Anthracnose":
        PLANT / "Mango Dataset" / "Anthracnose",
    "Mango___Bacterial_canker":
        PLANT / "Mango Dataset" / "Bacterial Canker",
    "Mango___Cutting_weevil":
        PLANT / "Mango Dataset" / "Cutting Weevil",
    "Mango___Die_back":
        PLANT / "Mango Dataset" / "die back",
    "Mango___Gall_midge":
        PLANT / "Mango Dataset" / "Gall Midge",
    "Mango___healthy":
        PLANT / "Mango Dataset" / "Healthy",
    "Mango___Powdery_mildew":
        PLANT / "Mango Dataset" / "Powdery Mildew",
    "Mango___Sooty_mould":
        PLANT / "Mango Dataset" / "Sooty Mould",

    # LUA NUOC
    "Rice___Bacterial_leaf_blight":
        PLANT / "rice_leaf_diseases"
        / "Bacterial leaf blight",
    "Rice___Brown_spot":
        PLANT / "rice_leaf_diseases" / "Brown spot",
    "Rice___Leaf_smut":
        PLANT / "rice_leaf_diseases" / "Leaf smut",

    # CAO SU - CHI DUNG RAW
    "Rubber___Anthracnose":
        PLANT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Anthracnose",
    "Rubber___Dry_leaf":
        PLANT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Dry_Leaf",
    "Rubber___healthy":
        PLANT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Healthy",
    "Rubber___Leaf_spot":
        PLANT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Leaf_Spot",
}

# 15 lop goc, chi them khi dung --include-old
OLD_CLASSES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]


def file_hash(path):
    """SHA256 de tim file trung noi dung byte."""
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def read_image(path):
    """
    Thu doc va giai ma toan bo anh.
    Neu Pillow loi, thu OpenCV neu da cai.
    Tra ve (RGB_image, method).
    """
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            img.load()
            return img.convert("RGB"), "Pillow"

    except Exception as pillow_error:
        try:
            import cv2
            import numpy as np

            raw = np.fromfile(str(path), dtype=np.uint8)
            decoded = cv2.imdecode(
                raw, cv2.IMREAD_COLOR
            )

            if decoded is None:
                raise ValueError("OpenCV khong giai ma duoc")

            rgb = cv2.cvtColor(
                decoded, cv2.COLOR_BGR2RGB
            )

            return Image.fromarray(rgb), "OpenCV"

        except Exception as cv_error:
            raise ValueError(
                f"Pillow: {type(pillow_error).__name__}: "
                f"{pillow_error}; "
                f"OpenCV: {type(cv_error).__name__}: "
                f"{cv_error}"
            ) from cv_error


def save_csv(path, rows, columns):
    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--include-old",
        action="store_true",
        help="Kiem tra ca 15 lop cu"
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Xuat anh JPG da giai ma sang thu muc moi"
    )

    args = parser.parse_args()

    folders = dict(CLASS_DIRS)

    if args.include_old:
        for name in OLD_CLASSES:
            folders[name] = PLANT / name

    REPORT.mkdir(parents=True, exist_ok=True)

    export_dir = None

    if args.export:
        stamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        export_dir = (
            ROOT / "dataset_exports" / f"dataset_{stamp}"
        )
        export_dir.mkdir(
            parents=True, exist_ok=False
        )

    valid_rows = []
    error_rows = []
    duplicate_rows = []
    summary_rows = []
    seen_hashes = {}

    print("=" * 70)
    print("KIEM TRA ANH THUC TE - PLANT DISEASE")
    print("ROOT:", ROOT)
    print("SO LOP CAN KIEM TRA:", len(folders))
    print("=" * 70)

    for class_name, folder in sorted(folders.items()):
        print(f"\n[{class_name}]")
        print("DUONG DAN:", folder)

        total = 0
        valid_count = 0
        error_count = 0
        duplicate_count = 0

        if not folder.is_dir():
            print("[KHONG TIM THAY THU MUC]")

            error_rows.append({
                "class_name": class_name,
                "source_path": str(folder),
                "error": "DIRECTORY_NOT_FOUND",
            })

            summary_rows.append({
                "class_name": class_name,
                "folder": str(folder),
                "status": "NOT_FOUND",
                "total": 0,
                "valid": 0,
                "errors": 0,
                "duplicates": 0,
            })
            continue

        images = sorted(
            (
                p for p in folder.rglob("*")
                if p.is_file()
                and p.suffix.lower() in EXTENSIONS
            ),
            key=lambda p: str(p).lower()
        )

        total = len(images)

        print("SO FILE ANH:", total)

        for index, path in enumerate(images, 1):
            if index % 250 == 0:
                print(
                    f"  Da kiem tra {index}/{total} anh",
                    flush=True
                )

            image = None

            try:
                image, method = read_image(path)
                width, height = image.size

                if width <= 0 or height <= 0:
                    raise ValueError("Kich thuoc anh khong hop le")

                digest = file_hash(path)

                if digest in seen_hashes:
                    duplicate_count += 1

                    original_class, original_path = (
                        seen_hashes[digest]
                    )

                    duplicate_rows.append({
                        "class_name": class_name,
                        "source_path": str(path),
                        "original_class": original_class,
                        "original_path": original_path,
                    })

                    continue

                # Chi ghi nhan hash sau khi anh hop le
                # va (neu export) da luu thanh cong.

                saved_path = str(path)

                if export_dir is not None:
                    output_folder = export_dir / class_name

                    output_folder.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    output_path = (
                        output_folder
                        / f"{digest[:32]}.jpg"
                    )

                    image.save(
                        output_path,
                        format="JPEG",
                        quality=92
                    )

                    saved_path = str(output_path)

                seen_hashes[digest] = (
                    class_name, str(path)
                )

                valid_rows.append({
                    "class_name": class_name,
                    "source_path": str(path),
                    "training_path": saved_path,
                    "width": width,
                    "height": height,
                    "decoded_by": method,
                    "sha256": digest,
                })

                valid_count += 1

            except Exception as exc:
                error_count += 1

                error_rows.append({
                    "class_name": class_name,
                    "source_path": str(path),
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                })

            finally:
                if image is not None:
                    image.close()

        status = "OK"

        if error_count or duplicate_count or total == 0:
            status = "WARNING"

        print(
            f"HOAN THANH: "
            f"Hop le={valid_count}, "
            f"Loi={error_count}, "
            f"Trung={duplicate_count}"
        )

        summary_rows.append({
            "class_name": class_name,
            "folder": str(folder),
            "status": status,
            "total": total,
            "valid": valid_count,
            "errors": error_count,
            "duplicates": duplicate_count,
        })

    save_csv(
        REPORT / "anh_hop_le.csv",
        valid_rows,
        [
            "class_name", "source_path",
            "training_path", "width", "height",
            "decoded_by", "sha256"
        ]
    )

    save_csv(
        REPORT / "anh_loi.csv",
        error_rows,
        ["class_name", "source_path", "error"]
    )

    save_csv(
        REPORT / "anh_trung.csv",
        duplicate_rows,
        [
            "class_name", "source_path",
            "original_class", "original_path"
        ]
    )

    save_csv(
        REPORT / "tong_hop.csv",
        summary_rows,
        [
            "class_name", "folder", "status",
            "total", "valid", "errors", "duplicates"
        ]
    )

    # Manifest de code training co the doc sau nay.
    save_csv(
        REPORT / "du_lieu_huan_luyen.csv",
        [
            {
                "image_path": row["training_path"],
                "label": row["class_name"],
            }
            for row in valid_rows
        ],
        ["image_path", "label"]
    )

    metadata = {
        "root": str(ROOT),
        "export_dir": (
            str(export_dir) if export_dir else None
        ),
        "class_names": sorted(
            row["class_name"]
            for row in summary_rows
            if row["status"] != "NOT_FOUND"
        ),
        "summary": summary_rows,
        "total_valid": len(valid_rows),
        "total_errors": len(error_rows),
        "total_duplicates": len(duplicate_rows),
    }

    (REPORT / "ket_qua.json").write_text(
        json.dumps(
            metadata, indent=2, ensure_ascii=False
        ),
        encoding="utf-8"
    )

    lines = [
        "BAO CAO KIEM TRA ANH THUC TE",
        "=" * 70,
    ]

    for row in summary_rows:
        lines.append(
            f"{row['class_name']} | {row['status']} | "
            f"Tong={row['total']} | "
            f"Hop le={row['valid']} | "
            f"Loi={row['errors']} | "
            f"Trung={row['duplicates']}"
        )
        lines.append(f"  {row['folder']}")

    lines += [
        "",
        f"ANH HOP LE: {len(valid_rows)}",
        f"FILE/LOI DUONG DAN: {len(error_rows)}",
        f"ANH TRUNG: {len(duplicate_rows)}",
        f"EXPORT: {export_dir or 'KHONG'}",
    ]

    (REPORT / "bao_cao.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("DA KIEM TRA XONG")
    print("Anh hop le:", len(valid_rows))
    print("Loi:", len(error_rows))
    print("Anh trung:", len(duplicate_rows))
    print("Bao cao:", REPORT)

    if export_dir:
        print("Anh duoc luu tai:", export_dir)

    if error_rows:
        print(
            "\n[CANH BAO] Co anh loi hoac "
            "thu muc khong ton tai."
        )

    if duplicate_rows:
        print(
            "[CANH BAO] Co anh trung noi dung. "
            "Xem anh_trung.csv."
        )


if __name__ == "__main__":
    main()
