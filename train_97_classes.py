
# -*- coding: utf-8 -*-
"""
HUAN LUYEN MOBILENETV2 - 97 LOP BENH LA CAY

Doc manifest da chia, khong quet/chia anh ngau nhien.
Khong ghi de mo hinh 41 lop.

Lenh:
  python train_97_classes.py --check_only
  python train_97_classes.py --epochs 1 --max_train_per_class 80
  python train_97_classes.py --epochs 8

Danh gia Test:
  python train_97_classes.py --test_checkpoint "DUONG_DAN_PTH"
"""

import os
import csv
import json
import math
import random
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

ROOT = Path(r"D:\DoanHMvUD")

os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache" / "matplotlib")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_architecture import get_mobilenet_v2_model


MANIFEST = (
    ROOT / "dataset_review"
    / "dataset_97_lop"
    / "split_97_lop"
    / "dataset_manifest_SPLIT_DU_KIEN.csv"
)

RUNS = ROOT / "training_runs"
VALID_SPLITS = {"train", "val", "test"}


def read_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    with MANIFEST.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Manifest rong!")

    required = {
        "path", "class", "split",
        "source", "group_id", "near_group_id"
    }

    if not required.issubset(rows[0]):
        raise ValueError("Manifest thieu cot bat buoc!")

    classes = sorted({
        r["class"] for r in rows
        if r["split"] in VALID_SPLITS
    })

    if len(classes) != 97:
        raise ValueError(
            f"Can 97 lop, tim thay {len(classes)} lop"
        )

    class_splits = defaultdict(Counter)
    groups = defaultdict(set)
    paths = set()
    review_count = Counter()

    for row in rows:
        path = row["path"]
        split = row["split"]
        label = row["class"]

        if path in paths:
            raise ValueError(
                "Duong dan trung trong manifest: " + path
            )
        paths.add(path)

        if split not in VALID_SPLITS:
            continue

        class_splits[label][split] += 1

        if row.get("needs_label_review") == "YES":
            review_count[split] += 1

        if row["source"] == "TEA_GROUPED":
            group = row["group_id"]
        else:
            group = row["near_group_id"]

        if group:
            groups[(label, group)].add(split)

    for label in classes:
        for split in VALID_SPLITS:
            if class_splits[label][split] == 0:
                raise ValueError(
                    f"Lop {label} thieu tap {split}"
                )

    leaking = [
        key for key, splits in groups.items()
        if len(splits) > 1
    ]

    if leaking:
        raise RuntimeError(
            f"Phat hien {len(leaking)} nhom "
            "nam trong nhieu tap!"
        )

    usable = [
        r for r in rows
        if r["split"] in VALID_SPLITS
    ]

    counts = Counter(r["split"] for r in usable)

    print("\n===== KIEM TRA MANIFEST =====")
    print("So lop:", len(classes))
    print("Train:", counts["train"])
    print("Validation:", counts["val"])
    print("Test:", counts["test"])
    print("Ro ri theo ID:", len(leaking))
    print("Anh can xem lai nhan:", sum(review_count.values()))
    print("Trong Test:", review_count["test"])

    small = [
        label for label in classes
        if class_splits[label]["test"] < 20
    ]

    print("Lop co duoi 20 anh Test:", len(small))

    return usable, classes


def prepare_train(rows, seed, max_per_class):
    rng = random.Random(seed)

    # Moi nhom la nguon Chi lay 1 bien the
    tea_groups = defaultdict(list)
    other_rows = []

    for row in rows:
        if row["split"] != "train":
            continue

        if row["source"] == "TEA_GROUPED":
            key = (row["class"], row["group_id"])
            tea_groups[key].append(row)
        else:
            other_rows.append(row)

    selected = other_rows.copy()

    for key in sorted(tea_groups):
        selected.append(rng.choice(tea_groups[key]))

    by_class = defaultdict(list)

    for row in selected:
        by_class[row["class"]].append(row)

    final = []

    for label in sorted(by_class):
        items = by_class[label]
        rng.shuffle(items)

        if max_per_class > 0:
            items = items[:max_per_class]

        final.extend(items)

    rng.shuffle(final)

    print("\n===== CAN BANG TRAIN =====")
    print("So nhom Tea trong Train:", len(tea_groups))
    print("Anh Train sau chon loc:", len(final))

    return final


class LeafDataset(Dataset):
    def __init__(self, rows, class_to_id, training=False):
        self.rows = rows
        self.class_to_id = class_to_id

        operations = [
            transforms.Resize((224, 224))
        ]

        if training:
            operations += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15
                )
            ]

        operations += [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]

        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["path"]) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            image = self.transform(img)

        label = self.class_to_id[row["class"]]

        return image, label, row["path"]


def make_loader(
    rows, class_to_id, batch_size,
    training=False, workers=0
):
    dataset = LeafDataset(
        rows, class_to_id, training=training
    )

    if training:
        counts = Counter(r["class"] for r in rows)

        # Lay mau can bang mem: 1 / sqrt(so anh)
        weights = [
            1.0 / math.sqrt(counts[r["class"]])
            for r in rows
        ]

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(
                weights, dtype=torch.double
            ),
            num_samples=len(rows),
            replacement=True
        )
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        n = labels.size(0)
        total += n
        total_loss += loss.item() * n

        correct += (
            logits.argmax(1) == labels
        ).sum().item()

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes):
    model.eval()

    losses = 0.0
    total = 0
    y_true = []
    y_pred = []
    paths = []

    for images, labels, batch_paths in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        predictions = logits.argmax(1)
        n = labels.size(0)

        total += n
        losses += loss.item() * n

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
        paths.extend(batch_paths)

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true, y_pred,
        labels=list(range(n_classes)),
        average="macro",
        zero_division=0
    )

    return {
        "loss": losses / total,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "y_true": y_true,
        "y_pred": y_pred,
        "paths": paths
    }


def save_test_report(result, classes, output):
    y_true = result["y_true"]
    y_pred = result["y_pred"]

    labels = list(range(len(classes)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        digits=4,
        zero_division=0
    )

    (output / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    with (output / "test_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "path", "actual", "predicted", "correct"
        ])

        for path, a, b in zip(
            result["paths"], y_true, y_pred
        ):
            writer.writerow([
                path, classes[a], classes[b], int(a == b)
            ])

    cm = confusion_matrix(
        y_true, y_pred, labels=labels
    )

    with (output / "confusion_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["actual/predicted"] + classes)

        for name, values in zip(classes, cm):
            writer.writerow([name] + list(values))

    fig, ax = plt.subplots(figsize=(18, 16))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xlabel("Predicted class ID")
    ax.set_ylabel("Actual class ID")
    ax.set_title("Confusion Matrix - 97 Classes")
    fig.tight_layout()
    fig.savefig(
        output / "confusion_matrix.png", dpi=150
    )
    plt.close(fig)

    summary = {
        "test_accuracy": result["accuracy"],
        "test_macro_f1": result["macro_f1"],
        "test_loss": result["loss"],
        "test_images": len(y_true),
        "classes": len(classes)
    }

    (output / "test_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n===== TEST RESULTS =====")
    print(
        f"Accuracy: {result['accuracy'] * 100:.2f}%"
    )
    print(
        f"Macro F1: {result['macro_f1']:.4f}"
    )
    print("Da luu bao cao tai:", output)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check_only", action="store_true"
    )
    parser.add_argument(
        "--test_checkpoint", type=str, default=""
    )
    parser.add_argument(
        "--epochs", type=int, default=8
    )
    parser.add_argument(
        "--head_epochs", type=int, default=1
    )
    parser.add_argument(
        "--batch_size", type=int, default=16
    )
    parser.add_argument(
        "--max_train_per_class", type=int, default=1200
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001
    )
    parser.add_argument(
        "--patience", type=int, default=3
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--workers", type=int, default=0
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows, classes = read_manifest()
    class_to_id = {
        name: i for i, name in enumerate(classes)
    }

    if args.check_only:
        print("\nKIEM TRA HOAN THANH. CHUA TRAIN.")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("\nThiet bi:", device)

    criterion = nn.CrossEntropyLoss()

    # Chi danh gia Test khi duoc yeu cau rieng
    if args.test_checkpoint:
        checkpoint_path = Path(args.test_checkpoint)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        model = get_mobilenet_v2_model(
            num_classes=len(classes),
            pretrained=False
        ).to(device)

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

        if checkpoint["class_names"] != classes:
            raise ValueError(
                "Thu tu nhan trong checkpoint "
                "khong khop manifest!"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        test_rows = [
            r for r in rows if r["split"] == "test"
        ]

        test_loader = make_loader(
            test_rows, class_to_id,
            args.batch_size, workers=args.workers
        )

        result = evaluate(
            model, test_loader, criterion,
            device, len(classes)
        )

        save_test_report(
            result, classes, checkpoint_path.parent
        )
        return

    train_rows = prepare_train(
        rows, args.seed, args.max_train_per_class
    )

    val_rows = [
        r for r in rows if r["split"] == "val"
    ]

    train_loader = make_loader(
        train_rows, class_to_id, args.batch_size,
        training=True, workers=args.workers
    )

    val_loader = make_loader(
        val_rows, class_to_id, args.batch_size,
        workers=args.workers
    )

    model = get_mobilenet_v2_model(
        num_classes=len(classes),
        pretrained=True
    ).to(device)

    # Fine-tuning sau giai doan hoc classifier
    for parameter in model.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2
    )

    run_name = datetime.now().strftime(
        "plant97_%Y%m%d_%H%M%S"
    )

    output = RUNS / run_name
    output.mkdir(parents=True, exist_ok=False)

    checkpoint_path = output / "best_plant_model.pth"

    (output / "class_names.json").write_text(
        json.dumps(
            classes, ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )

    (output / "training_config.json").write_text(
        json.dumps(
            vars(args), indent=2, ensure_ascii=False
        ),
        encoding="utf-8"
    )

    history = []
    best_f1 = -1.0
    bad_epochs = 0

    print("\n===== BAT DAU TRAIN 97 LOP =====")

    for epoch in range(args.epochs):

        # Epoch dau: hoc bo phan phan loai
        freeze = epoch < args.head_epochs

        if hasattr(model, "features"):
            model.features.requires_grad_(not freeze)

        train_loss, train_acc = train_epoch(
            model, train_loader,
            optimizer, criterion, device
        )

        val = evaluate(
            model, val_loader,
            criterion, device, len(classes)
        )

        scheduler.step(val["macro_f1"])

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["accuracy"],
            "val_macro_f1": val["macro_f1"]
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train: {train_acc:.2%} | "
            f"Val: {val['accuracy']:.2%} | "
            f"Macro F1: {val['macro_f1']:.4f}",
            flush=True
        )

        if val["macro_f1"] > best_f1 + 1e-5:
            best_f1 = val["macro_f1"]
            bad_epochs = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": classes,
                "val_macro_f1": best_f1,
                "epoch": epoch + 1
            }, checkpoint_path)

            print("  => Da luu model tot nhat")
        else:
            bad_epochs += 1

        with (output / "history.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(history[0])
            )
            writer.writeheader()
            writer.writerows(history)

        if bad_epochs >= args.patience:
            print("Early stopping!")
            break

    print("\n===== HOAN THANH HUAN LUYEN =====")
    print("Best Validation Macro F1:", round(best_f1, 4))
    print("Model:", checkpoint_path)
    print("Chua danh gia tren Test.")


if __name__ == "__main__":
    main()

#Bước 3 python -u train_97_classes.py --epochs 3 --head_epochs 1 --max_train_per_class 100 --batch_size 8