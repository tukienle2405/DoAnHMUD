
# -*- coding: utf-8 -*-
"""
TRAIN MOBILE NET V2 - BENH LA CAY
15 lop goc + 26 lop moi = 41 lop

Khong di chuyen anh.
Khong ghi de mo hinh 15 lop cu.
"""

import os
import sys
import json
import time
import random
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

ROOT = Path(r"D:\DoanHMvUD")
PLANT = ROOT / "PlantVillage"

os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(
        encoding="utf-8",
        line_buffering=True
    )

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from model_architecture import get_mobilenet_v2_model


# ===========================================
# 1. DANH SACH 15 LOP CU
# ===========================================

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


# ===========================================
# 2. DANH SACH CAC DATASET MOI
# ===========================================

NEW_DATASETS = {
    "Chili": (
        "chileleaf",
        {
            "Bacterial_Spot": "Chili___Bacterial_spot",
            "Cercospora_Leaf_Spot": "Chili___Cercospora_leaf_spot",
            "Curl_Virus": "Chili___Curl_virus",
            "Healthy_Leaf": "Chili___healthy",
        }
    ),

    "Corn": (
        "corndisease",
        {
            "Blight": "Corn___Blight",
            "Common_Rust": "Corn___Common_rust",
            "Gray_Leaf_Spot": "Corn___Gray_leaf_spot",
            "Healthy": "Corn___healthy",
        }
    ),

    "Coffee": (
        "A Machine  Learning Dataset for Classification "
        "of Common Coffee Leaf Diseases in Uganda/coffee dataset",
        {
            "Health leaves": "Coffee___healthy",
            "leaf rust": "Coffee___Rust",
            "phoma": "Coffee___Phoma",
        }
    ),

    "Rubber": (
        "Rubber_Leaf_Dataset/Raw_Dataset",
        {
            "Anthracnose": "Rubber___Anthracnose",
            "Dry_Leaf": "Rubber___Dry_leaf",
            "Healthy": "Rubber___healthy",
            "Leaf_Spot": "Rubber___Leaf_spot",
        }
    ),

    "Mango": (
        "Mango Dataset",
        {
            "Anthracnose": "Mango___Anthracnose",
            "Bacterial Canker": "Mango___Bacterial_canker",
            "Cutting Weevil": "Mango___Cutting_weevil",
            "die back": "Mango___Die_back",
            "Gall Midge": "Mango___Gall_midge",
            "Healthy": "Mango___healthy",
            "Powdery Mildew": "Mango___Powdery_mildew",
            "Sooty Mould": "Mango___Sooty_mould",
        }
    ),

    "Rice": (
        "rice_leaf_diseases",
        {
            "Bacterial leaf blight": "Rice___Bacterial_leaf_blight",
            "Brown spot": "Rice___Brown_spot",
            "Leaf smut": "Rice___Leaf_smut",
        }
    ),
}

EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
    ".tif", ".tiff"
}


# ===========================================
# 3. XAY DUNG DUONG DAN CAC LOP
# ===========================================

def get_class_folders():
    folders = {}

    for name in OLD_CLASSES:
        folders[name] = PLANT / name

    for crop, (relative_path, classes) in NEW_DATASETS.items():
        base = PLANT / relative_path

        for original_name, class_name in classes.items():
            if class_name in folders:
                raise ValueError(
                    f"Trung ten lop: {class_name}"
                )

            folders[class_name] = base / original_name

    return dict(sorted(folders.items()))


def scan_dataset():
    class_folders = get_class_folders()
    found = {}
    problems = []

    print("\n===== KIEM TRA DU LIEU =====")
    print("Duong dan:", PLANT)

    for name, folder in class_folders.items():
        if not folder.is_dir():
            problems.append(
                f"Khong tim thay: {name} => {folder}"
            )
            continue

        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        )

        found[name] = images

        print(f"{name}: {len(images)} anh")

        if len(images) < 20:
            problems.append(
                f"Lop {name} chi co {len(images)} anh"
            )

    print(f"\nSo lop tim thay: {len(found)}")
    print(
        "Tong so anh:",
        sum(len(v) for v in found.values())
    )

    if problems:
        print("\nCAC VAN DE:")
        for problem in problems:
            print("-", problem)

        raise ValueError(
            "Du lieu chua day du. Hay kiem tra duong dan."
        )

    if len(found) != 41:
        raise ValueError(
            f"Can 41 lop, hien co {len(found)} lop"
        )

    return found


# ===========================================
# 4. KIEM TRA ANH VA LOAI BO ANH TRUNG
# ===========================================

def hash_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def prepare_samples(folders, max_samples, seed):
    rng = random.Random(seed)
    selected = {}
    hashes = {}
    duplicates = 0
    broken = 0

    for name, paths in folders.items():
        shuffled = paths.copy()
        rng.shuffle(shuffled)

        chosen = []

        for path in shuffled:
            try:
                with Image.open(path) as image:
                    image.verify()

                digest = hash_file(path)

            except Exception as exc:
                broken += 1
                print(f"[ANH LOI] {path}: {exc}")
                continue

            if digest in hashes:
                duplicates += 1

                if hashes[digest] != name:
                    print(
                        "[CANH BAO NHAN TRUNG]",
                        name, "<=>", hashes[digest]
                    )

                continue

            hashes[digest] = name
            chosen.append(path)

            if max_samples > 0:
                if len(chosen) >= max_samples:
                    break

        if len(chosen) < 20:
            raise ValueError(
                f"Lop {name} chi con {len(chosen)} "
                "anh hop le sau kiem tra."
            )

        selected[name] = chosen

    print("\nAnh trung hoan toan:", duplicates)
    print("Anh khong doc duoc:", broken)

    return selected


# ===========================================
# 5. CHIA TRAIN / VALIDATION / TEST
# ===========================================

def split_data(selected, class_names, seed):
    rng = random.Random(seed)

    train_rows = []
    val_rows = []
    test_rows = []

    for label, name in enumerate(class_names):
        images = selected[name].copy()
        rng.shuffle(images)

        n = len(images)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))

        val_images = images[:n_val]
        test_images = images[n_val:n_val + n_test]
        train_images = images[n_val + n_test:]

        train_rows.extend((str(p), label) for p in train_images)
        val_rows.extend((str(p), label) for p in val_images)
        test_rows.extend((str(p), label) for p in test_images)

    print("\n===== PHAN CHIA DU LIEU =====")
    print("Train:", len(train_rows))
    print("Validation:", len(val_rows))
    print("Test:", len(test_rows))

    return train_rows, val_rows, test_rows


# ===========================================
# 6. DATASET VA TIEN XU LY
# ===========================================

class PlantDataset(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows

        operations = [
            transforms.Resize((224, 224))
        ]

        if training:
            operations.extend([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.2),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15
                ),
            ])

        operations.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, label = self.rows[index]

        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)

        return tensor, label


# ===========================================
# 7. HUAN LUYEN MOT EPOCH
# ===========================================

def train_one_epoch(
    model, loader, criterion, optimizer, device
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        n = images.size(0)

        total_loss += loss.item() * n
        correct += (
            outputs.argmax(dim=1) == labels
        ).sum().item()
        total += n

    return total_loss / total, correct / total


# ===========================================
# 8. DANH GIA MO HINH
# ===========================================

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            n = images.size(0)
            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * n
            correct += (
                predictions == labels
            ).sum().item()
            total += n

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return (
        total_loss / total,
        correct / total,
        y_true,
        y_pred
    )


# ===========================================
# 9. LUU BIEU DO VA BAO CAO
# ===========================================

def save_evaluation(
    model, loader, criterion,
    device, classes, history, output_dir
):
    loss, accuracy, y_true, y_pred = evaluate(
        model, loader, criterion, device
    )

    print(f"\nTEST ACCURACY: {accuracy * 100:.2f}%")

    labels = list(range(len(classes)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        zero_division=0,
        digits=4
    )

    print(report)

    (output_dir / "classification_report.txt").write_text(
        f"Test Accuracy: {accuracy:.4f}\n"
        f"Test Loss: {loss:.4f}\n\n"
        + report,
        encoding="utf-8"
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=labels
    )

    fig, ax = plt.subplots(figsize=(25, 25))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        include_values=False,
        colorbar=False
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=180
    )
    plt.close(fig)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "training_curves.png",
        dpi=180
    )
    plt.close(fig)


# ===========================================
# 10. CHUONG TRINH CHINH
# ===========================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_only", action="store_true")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    folders = scan_dataset()

    if args.check_only:
        print("\nKIEM TRA THU MUC HOAN THANH.")
        print("Chua train mo hinh.")
        return

    selected = prepare_samples(
        folders, args.samples, args.seed
    )

    class_names = sorted(selected)

    train_rows, val_rows, test_rows = split_data(
        selected, class_names, args.seed
    )

    train_loader = DataLoader(
        PlantDataset(train_rows, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    val_loader = DataLoader(
        PlantDataset(val_rows),
        batch_size=args.batch_size,
        num_workers=0
    )

    test_loader = DataLoader(
        PlantDataset(test_rows),
        batch_size=args.batch_size,
        num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nThiet bi:", device)
    print("So lop:", len(class_names))

    model = get_mobilenet_v2_model(
        num_classes=len(class_names),
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    run_name = datetime.now().strftime(
        "plant_%Y%m%d_%H%M%S"
    )

    output_dir = ROOT / "training_runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    checkpoint_path = output_dir / "best_plant_model.pth"

    (output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0

    print("\n===== BAT DAU HUAN LUYEN =====")

    start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader,
            criterion, optimizer, device
        )

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        for key, value in [
            ("train_loss", train_loss),
            ("train_acc", train_acc),
            ("val_loss", val_loss),
            ("val_acc", val_acc),
        ]:
            history[key].append(value)

        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"({elapsed:.1f}s) | "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "val_acc": val_acc,
                "history": history,
            }, checkpoint_path)

            print("  => Da luu checkpoint tot nhat")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    save_evaluation(
        model, test_loader,
        criterion, device,
        class_names, history, output_dir
    )

    print("\n===== HOAN THANH =====")
    print(
        f"Tong thoi gian: {(time.time() - start) / 60:.2f} phut"
    )
    print("Thu muc ket qua:", output_dir)


if __name__ == "__main__":
    main()
    