
# -*- coding: utf-8 -*-
"""
DANH GIA TOAN BO DU LIEU BANG MOBILENETV2

Chay thu:
    python evaluate_all_images.py --limit 500

Chay toan bo:
    python evaluate_all_images.py

Khong huan luyen lai mo hinh.
Khong thay doi checkpoint goc.
"""

import argparse
import csv
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from model_architecture import get_mobilenet_v2_model


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "PlantVillage"
RUN_ROOT = ROOT / "training_runs"
OUTPUT_ROOT = ROOT / "evaluation_all_runs"

EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}

CROP_NAMES = {
    "Chili": "Ớt cay",
    "Coffee": "Cà phê",
    "Corn": "Ngô",
    "Mango": "Xoài",
    "Pepper": "Ớt chuông",
    "Potato": "Khoai tây",
    "Rice": "Lúa nước",
    "Rubber": "Cao su",
    "Tomato": "Cà chua",
}

COFFEE_ROOT = (
    DATA_ROOT
    / "A Machine  Learning Dataset for Classification "
      "of Common Coffee Leaf Diseases in Uganda"
    / "coffee dataset"
)

# Anh cua 26 lop moi nam trong cac dataset rieng
NEW_FOLDERS = {
    "Chili___Bacterial_spot":
        DATA_ROOT / "chileleaf" / "Bacterial_Spot",
    "Chili___Cercospora_leaf_spot":
        DATA_ROOT / "chileleaf" / "Cercospora_Leaf_Spot",
    "Chili___Curl_virus":
        DATA_ROOT / "chileleaf" / "Curl_Virus",
    "Chili___healthy":
        DATA_ROOT / "chileleaf" / "Healthy_Leaf",

    "Coffee___Phoma":
        COFFEE_ROOT / "phoma",
    "Coffee___Rust":
        COFFEE_ROOT / "leaf rust",
    "Coffee___healthy":
        COFFEE_ROOT / "Health leaves",

    "Corn___Blight":
        DATA_ROOT / "corndisease" / "Blight",
    "Corn___Common_rust":
        DATA_ROOT / "corndisease" / "Common_Rust",
    "Corn___Gray_leaf_spot":
        DATA_ROOT / "corndisease" / "Gray_Leaf_Spot",
    "Corn___healthy":
        DATA_ROOT / "corndisease" / "Healthy",

    "Mango___Anthracnose":
        DATA_ROOT / "Mango Dataset" / "Anthracnose",
    "Mango___Bacterial_canker":
        DATA_ROOT / "Mango Dataset" / "Bacterial Canker",
    "Mango___Cutting_weevil":
        DATA_ROOT / "Mango Dataset" / "Cutting Weevil",
    "Mango___Die_back":
        DATA_ROOT / "Mango Dataset" / "die back",
    "Mango___Gall_midge":
        DATA_ROOT / "Mango Dataset" / "Gall Midge",
    "Mango___Powdery_mildew":
        DATA_ROOT / "Mango Dataset" / "Powdery Mildew",
    "Mango___Sooty_mould":
        DATA_ROOT / "Mango Dataset" / "Sooty Mould",
    "Mango___healthy":
        DATA_ROOT / "Mango Dataset" / "Healthy",

    "Rice___Bacterial_leaf_blight":
        DATA_ROOT / "rice_leaf_diseases"
        / "Bacterial leaf blight",
    "Rice___Brown_spot":
        DATA_ROOT / "rice_leaf_diseases" / "Brown spot",
    "Rice___Leaf_smut":
        DATA_ROOT / "rice_leaf_diseases" / "Leaf smut",

    "Rubber___Anthracnose":
        DATA_ROOT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Anthracnose",
    "Rubber___Dry_leaf":
        DATA_ROOT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Dry_Leaf",
    "Rubber___Leaf_spot":
        DATA_ROOT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Leaf_Spot",
    "Rubber___healthy":
        DATA_ROOT / "Rubber_Leaf_Dataset"
        / "Raw_Dataset" / "Healthy",
}


def get_crop(class_name):
    for prefix, crop in CROP_NAMES.items():
        if class_name.startswith(prefix):
            return crop
    return "Khác"


def load_checkpoint_file(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )
    except TypeError:
        # Phiên bản PyTorch cũ không có weights_only
        return torch.load(path, map_location="cpu")


def find_checkpoint(manual_path=None):
    if manual_path:
        path = Path(manual_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        checkpoint = load_checkpoint_file(path)
        return path, checkpoint

    candidates = []

    root_model = ROOT / "best_plant_model.pth"
    if root_model.is_file():
        candidates.append(root_model)

    if RUN_ROOT.is_dir():
        candidates.extend(RUN_ROOT.rglob("*.pth"))
        candidates.extend(RUN_ROOT.rglob("*.pt"))

    candidates = sorted(
        set(candidates),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    # Uu tien checkpoint DU 41 LOP,
    # khong vo tinh chon mo hinh 15 lop cu.
    for path in candidates:
        try:
            checkpoint = load_checkpoint_file(path)

            if not isinstance(checkpoint, dict):
                continue

            names = checkpoint.get("class_names", [])

            if (
                len(names) == 41
                and "model_state_dict" in checkpoint
            ):
                return path, checkpoint

        except Exception as exc:
            print(
                f"[BO QUA CHECKPOINT] {path.name}: {exc}"
            )

    raise RuntimeError(
        "Khong tim thay checkpoint 41 lop. "
        "Hay kiem tra training_runs."
    )


def get_class_folder(class_name):
    if class_name in NEW_FOLDERS:
        return NEW_FOLDERS[class_name]

    # 15 lop PlantVillage ban dau
    return DATA_ROOT / class_name


def collect_images(class_names):
    records = []
    missing = []

    print("\n=== KIEM TRA 41 THU MUC DU LIEU ===")

    for class_index, class_name in enumerate(class_names):
        folder = get_class_folder(class_name)

        if not folder.is_dir():
            missing.append((class_name, str(folder)))
            print(f"[THIEU] {class_name}")
            continue

        paths = sorted(
            p for p in folder.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower() in EXTENSIONS
            )
        )

        print(f"{class_name}: {len(paths):,} anh")

        for path in paths:
            records.append(
                (str(path), class_index)
            )

    if missing:
        lines = [
            f"{name}: {path}"
            for name, path in missing
        ]

        raise RuntimeError(
            "Co thu muc du lieu khong ton tai:\n"
            + "\n".join(lines)
        )

    return records


class PlantImageDataset(Dataset):
    def __init__(self, records):
        self.records = records

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label = self.records[index]

        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                tensor = self.transform(image)

            return tensor, label, path, None

        except Exception as exc:
            return None, label, path, str(exc)


def safe_collate(batch):
    valid = []
    errors = []

    for image, label, path, error in batch:
        if image is None:
            errors.append((path, error))
        else:
            valid.append((image, label, path))

    if not valid:
        return None, [], [], errors

    images = torch.stack(
        [item[0] for item in valid]
    )

    labels = [item[1] for item in valid]
    paths = [item[2] for item in valid]

    return images, labels, paths, errors


def write_csv(path, headers, rows):
    # utf-8-sig de Excel hien tieng Viet dung
    with open(
        path, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def save_confusion_plot(matrix, class_names, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(20, 19))

        image = ax.imshow(
            matrix,
            interpolation="nearest",
            cmap="Blues"
        )

        ax.set_xticks(
            np.arange(len(class_names))
        )
        ax.set_yticks(
            np.arange(len(class_names))
        )

        ax.set_xticklabels(
            class_names,
            rotation=90,
            fontsize=6
        )
        ax.set_yticklabels(
            class_names,
            fontsize=6
        )

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(
            "Confusion Matrix - MobileNetV2"
        )

        fig.colorbar(
            image, ax=ax, fraction=0.025
        )

        fig.subplots_adjust(
            left=0.34,
            bottom=0.34,
            right=0.95,
            top=0.95
        )

        fig.savefig(
            output,
            dpi=170,
            bbox_inches="tight"
        )
        plt.close(fig)

    except Exception as exc:
        print(
            "[CANH BAO] Khong tao duoc PNG:",
            exc
        )


def evaluate(args):
    print("\n" + "=" * 65)
    print("DANH GIA TOAN BO ANH - MOBILENETV2")
    print("=" * 65)

    checkpoint_path, checkpoint = find_checkpoint(
        args.checkpoint
    )

    class_names = list(
        checkpoint.get("class_names", [])
    )

    if len(class_names) != 41:
        raise RuntimeError(
            f"Checkpoint chi co {len(class_names)} lop, "
            "khong phai 41 lop."
        )

    print("\nCheckpoint:", checkpoint_path)
    print("So lop:", len(class_names))

    device = torch.device(
        "cuda"
        if (
            args.device == "auto"
            and torch.cuda.is_available()
        )
        else (
            args.device
            if args.device != "auto"
            else "cpu"
        )
    )

    print("Thiet bi:", device)

    model = get_mobilenet_v2_model(
        num_classes=len(class_names),
        pretrained=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    records = collect_images(class_names)

    original_count = len(records)

    if args.limit > 0:
        random.Random(42).shuffle(records)
        records = records[:args.limit]

    print("\nTong file anh tim thay:", original_count)
    print("So anh du kien danh gia:", len(records))

    if not records:
        raise RuntimeError("Khong tim thay anh nao.")

    mode = (
        "benchmark"
        if args.limit > 0
        else "full"
    )

    run_name = (
        f"{mode}_"
        + datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output_dir = OUTPUT_ROOT / run_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    dataset = PlantImageDataset(records)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=safe_collate,
        pin_memory=(device.type == "cuda")
    )

    true_labels = []
    pred_labels = []
    details = []
    errors = []

    start = time.perf_counter()
    processed = 0

    print("\n=== BAT DAU DANH GIA ===")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, 1):
            images, labels, paths, batch_errors = batch

            errors.extend(batch_errors)

            if images is not None:
                images = images.to(device)

                outputs = model(images)

                probabilities = torch.softmax(
                    outputs, dim=1
                )

                best_prob, predictions = (
                    probabilities.max(dim=1)
                )

                predictions = (
                    predictions.cpu().tolist()
                )

                best_prob = (
                    best_prob.cpu().tolist()
                )

                for i, actual in enumerate(labels):
                    predicted = predictions[i]

                    actual_name = class_names[actual]
                    pred_name = class_names[predicted]

                    is_correct = int(
                        actual == predicted
                    )

                    true_labels.append(actual)
                    pred_labels.append(predicted)

                    details.append([
                        paths[i],
                        get_crop(actual_name),
                        actual_name,
                        pred_name,
                        get_crop(pred_name),
                        is_correct,
                        round(
                            best_prob[i] * 100, 4
                        )
                    ])

                processed += len(labels)

            if (
                batch_idx % 20 == 0
                or batch_idx == len(loader)
            ):
                elapsed = (
                    time.perf_counter() - start
                )

                completed = (
                    processed + len(errors)
                )

                speed = (
                    completed / elapsed
                    if elapsed > 0 else 0
                )

                remaining = (
                    len(records) - completed
                )

                eta = (
                    remaining / speed / 60
                    if speed > 0 else 0
                )

                print(
                    f"[{completed:,}/{len(records):,}] "
                    f"Toc do: {speed:.1f} anh/s | "
                    f"Con khoang: {eta:.1f} phut",
                    flush=True
                )

    elapsed = time.perf_counter() - start

    if not true_labels:
        raise RuntimeError(
            "Khong doc duoc anh hop le nao."
        )

    total = len(true_labels)
    correct = sum(
        a == b
        for a, b in zip(
            true_labels, pred_labels
        )
    )

    incorrect = total - correct

    accuracy = accuracy_score(
        true_labels, pred_labels
    )

    labels = list(
        range(len(class_names))
    )

    report = classification_report(
        true_labels,
        pred_labels,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            pred_labels,
            average="weighted",
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        true_labels,
        pred_labels,
        labels=labels
    )

    # --------------------------------------------------------
    # LUU DANH SACH ANH DUNG/SAI
    # --------------------------------------------------------

    headers = [
        "image_path",
        "actual_crop",
        "actual_class",
        "predicted_class",
        "predicted_crop",
        "is_correct",
        "confidence_percent"
    ]

    write_csv(
        output_dir / "du_doan_chi_tiet.csv",
        headers,
        details
    )

    wrong_rows = [
        row for row in details
        if row[5] == 0
    ]

    write_csv(
        output_dir / "anh_du_doan_sai.csv",
        headers,
        wrong_rows
    )

    write_csv(
        output_dir / "anh_khong_doc_duoc.csv",
        ["image_path", "error"],
        errors
    )

    # --------------------------------------------------------
    # THONG KE THEO TUNG LOAI CAY
    # --------------------------------------------------------

    crop_stats = defaultdict(
        lambda: [0, 0]
    )

    for actual, predicted in zip(
        true_labels, pred_labels
    ):
        crop = get_crop(
            class_names[actual]
        )

        crop_stats[crop][0] += 1

        if actual == predicted:
            crop_stats[crop][1] += 1

    crop_rows = []

    for crop in sorted(crop_stats):
        n, right = crop_stats[crop]

        crop_rows.append([
            crop,
            n,
            right,
            n - right,
            round(right / n * 100, 2)
        ])

    write_csv(
        output_dir / "thong_ke_theo_cay.csv",
        [
            "crop",
            "total",
            "correct",
            "incorrect",
            "accuracy_percent"
        ],
        crop_rows
    )

    # --------------------------------------------------------
    # THONG KE THEO TUNG LOP
    # --------------------------------------------------------

    class_rows = []

    for idx, class_name in enumerate(class_names):
        n = sum(
            actual == idx
            for actual in true_labels
        )

        right = sum(
            actual == idx and predicted == idx
            for actual, predicted in zip(
                true_labels, pred_labels
            )
        )

        class_rows.append([
            get_crop(class_name),
            class_name,
            n,
            right,
            n - right,
            round(
                right / n * 100, 2
            ) if n else 0
        ])

    write_csv(
        output_dir / "thong_ke_41_lop.csv",
        [
            "crop",
            "class",
            "total",
            "correct",
            "incorrect",
            "accuracy_percent"
        ],
        class_rows
    )

    # --------------------------------------------------------
    # MA TRAN NHAM LAN
    # --------------------------------------------------------

    matrix_rows = []

    for idx, row in enumerate(matrix):
        matrix_rows.append([
            class_names[idx],
            *row.tolist()
        ])

    write_csv(
        output_dir / "confusion_matrix.csv",
        ["actual_class", *class_names],
        matrix_rows
    )

    save_confusion_plot(
        matrix,
        class_names,
        output_dir / "confusion_matrix.png"
    )

    # --------------------------------------------------------
    # BAO CAO DANH GIA
    # --------------------------------------------------------

    summary = f"""
DANH GIA MOBILENETV2 TREN TOAN BO DU LIEU
========================================

Checkpoint: {checkpoint_path}

So lop: {len(class_names)}
Tong file anh tim thay: {original_count}
So anh duoc chon: {len(records)}
So anh doc thanh cong: {total}
So anh loi: {len(errors)}

Anh du doan dung: {correct}
Anh du doan sai: {incorrect}

Accuracy: {accuracy:.6f}
Accuracy (%): {accuracy * 100:.2f}%
Weighted Precision: {precision:.6f}
Weighted Recall: {recall:.6f}
Weighted F1: {f1:.6f}

Tong thoi gian: {elapsed:.2f} giay
Toc do trung binh: {total / elapsed:.2f} anh/giay

LUU Y:
Day la danh gia tren du lieu bao gom ca
anh tung duoc dung de huan luyen.
Khong su dung Accuracy nay de thay the
Test Accuracy tren tap Test doc lap.

========================================
CLASSIFICATION REPORT
========================================

{report}
"""

    (output_dir / "bao_cao_tong_hop.txt").write_text(
        summary,
        encoding="utf-8"
    )

    (output_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("KET QUA CUOI CUNG")
    print("=" * 65)

    print("Tong anh doc thanh cong:", total)
    print("Anh du doan dung:", correct)
    print("Anh du doan sai:", incorrect)
    print("Anh khong doc duoc:", len(errors))

    print(f"ACCURACY: {accuracy * 100:.2f}%")
    print(f"PRECISION: {precision * 100:.2f}%")
    print(f"RECALL: {recall * 100:.2f}%")
    print(f"F1-SCORE: {f1 * 100:.2f}%")

    print(f"Thoi gian: {elapsed / 60:.2f} phut")
    print("\nBao cao da luu tai:")
    print(output_dir)
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="So anh danh gia; 0 la tat ca"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error(
            "--batch-size phai >= 1"
        )

    evaluate(args)
