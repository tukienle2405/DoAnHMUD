
# -*- coding: utf-8 -*-
"""
MobileNetV2 - Full training 97 classes
- Su dung tat ca anh Train
- Khong dung Test khi train
- Khoi tao tu checkpoint 97 lop gan nhat
- AMP cho NVIDIA GPU
- Luu checkpoint theo Validation Macro F1
"""

import os
import csv
import json
import math
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

ROOT = Path(r"D:\DoanHMvUD")
os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from model_architecture import get_mobilenet_v2_model

MANIFEST = (
    ROOT / "dataset_review"
    / "dataset_97_lop"
    / "split_97_lop"
    / "dataset_manifest_SPLIT_DU_KIEN.csv"
)

RUNS = ROOT / "training_runs"


class PlantDataset(Dataset):
    def __init__(self, rows, class_to_id, training=False):
        self.rows = rows
        self.class_to_id = class_to_id

        operations = [transforms.Resize((224, 224))]

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

        return image, self.class_to_id[row["class"]]


def load_data():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    with MANIFEST.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    classes = sorted({
        r["class"] for r in rows
        if r["split"] in {"train", "val", "test"}
    })

    if len(classes) != 97:
        raise ValueError(
            f"Can 97 lop, nhung tim thay {len(classes)}"
        )

    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test_count = sum(r["split"] == "test" for r in rows)

    counts = Counter(r["class"] for r in train)

    for name in classes:
        if counts[name] == 0:
            raise ValueError(f"Lop khong co anh Train: {name}")

    # Dam bao nhom anh khong trai qua nhieu tap
    group_splits = {}

    for r in rows:
        split = r["split"]
        if split not in {"train", "val", "test"}:
            continue

        if r["source"] == "TEA_GROUPED":
            group = r["group_id"]
        else:
            group = r["near_group_id"]

        if not group:
            continue

        key = (r["class"], group)
        previous = group_splits.setdefault(key, split)

        if previous != split:
            raise ValueError(
                f"Roi ri nhom: {key}, {previous}, {split}"
            )

    print("\n===== DATASET 97 LOP =====")
    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test giu rieng:", test_count)
    print("So lop:", len(classes))

    return train, val, classes, counts


def get_initial_checkpoint(classes, argument):
    if argument.lower() == "none":
        return None

    if argument.lower() == "auto":
        candidates = list(
            RUNS.glob("plant97_*/best_plant_model.pth")
        )

        if not candidates:
            raise FileNotFoundError(
                "Khong tim thay checkpoint plant97!"
            )

        path = max(
            candidates,
            key=lambda p: p.stat().st_mtime
        )
    else:
        path = Path(argument)

    if not path.is_file():
        raise FileNotFoundError(path)

    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True
    )

    if checkpoint["class_names"] != classes:
        raise ValueError(
            "Danh sach 97 lop trong checkpoint "
            "khong khop manifest!"
        )

    return path, checkpoint


def train_epoch(
    model, loader, optimizer, criterion,
    scaler, device, amp
):
    model.train()

    total = 0
    correct = 0
    loss_sum = 0.0

    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(
            device, non_blocking=True
        )
        labels = labels.to(
            device, non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            enabled=amp
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        n = labels.size(0)
        total += n

        loss_sum += loss.item() * n
        correct += (
            outputs.argmax(1) == labels
        ).sum().item()

        if step % 400 == 0 or step == len(loader):
            print(
                f"Train: {step}/{len(loader)} batch "
                f"({100 * step / len(loader):.1f}%)",
                flush=True
            )

    return loss_sum / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device, amp, n_classes):
    model.eval()

    loss_sum = 0.0
    total = 0
    actual = []
    predicted = []

    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(
            device, non_blocking=True
        )
        labels = labels.to(
            device, non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp
        ):
            outputs = model(images)
            loss = criterion(outputs, labels)

        predictions = outputs.argmax(1)

        n = labels.size(0)
        total += n
        loss_sum += loss.item() * n

        actual.extend(labels.cpu().tolist())
        predicted.extend(predictions.cpu().tolist())

        if step % 400 == 0 or step == len(loader):
            print(
                f"Validation: {step}/{len(loader)} batch "
                f"({100 * step / len(loader):.1f}%)",
                flush=True
            )

    accuracy = np.mean(
        np.asarray(actual) == np.asarray(predicted)
    )

    macro_f1 = f1_score(
        actual, predicted,
        labels=list(range(n_classes)),
        average="macro",
        zero_division=0
    )

    return (
        loss_sum / total,
        float(accuracy),
        float(macro_f1)
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default="auto"
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    amp = device.type == "cuda"

    print("\nThiet bi:", device)
    print("AMP:", amp)

    train_rows, val_rows, classes, counts = load_data()

    class_to_id = {
        name: i for i, name in enumerate(classes)
    }

    train_loader = DataLoader(
        PlantDataset(train_rows, class_to_id, True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=amp,
        drop_last=True
    )

    val_loader = DataLoader(
        PlantDataset(val_rows, class_to_id, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=amp
    )

    print("Batch Train moi epoch:", len(train_loader))
    print("Batch Validation:", len(val_loader))

    initial = get_initial_checkpoint(
        classes, args.init_checkpoint
    )

    model = get_mobilenet_v2_model(
        num_classes=len(classes),
        pretrained=(initial is None)
    ).to(device)

    model.requires_grad_(True)

    if initial is not None:
        path, checkpoint = initial
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        print("Khoi tao tu:", path)
    else:
        print("Khoi tao tu pretrained ImageNet")

    # Can bang vua phai thong qua loss.
    # Khong lay mau co hoan lai: moi anh Train
    # duoc doc 1 lan trong moi epoch.
    values = sorted(counts.values())
    median = values[len(values) // 2]

    weights = [
        min(
            3.0,
            max(0.5, math.sqrt(median / counts[name]))
        )
        for name in classes
    ]

    train_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights, dtype=torch.float32, device=device
        )
    )

    val_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp
    )

    run_name = datetime.now().strftime(
        "plant97_full_%Y%m%d_%H%M%S"
    )

    output = RUNS / run_name
    output.mkdir(parents=True, exist_ok=False)

    (output / "class_names.json").write_text(
        json.dumps(
            classes, ensure_ascii=False, indent=2
        ),
        encoding="utf-8"
    )

    config = dict(vars(args))
    config.update({
        "device": str(device),
        "amp": amp,
        "train_images": len(train_rows),
        "validation_images": len(val_rows),
        "initial_checkpoint": (
            str(initial[0]) if initial else None
        ),
        "manifest": str(MANIFEST)
    })

    (output / "training_config.json").write_text(
        json.dumps(
            config, indent=2, ensure_ascii=False
        ),
        encoding="utf-8"
    )

    history = []
    best_f1 = -1.0
    bad_epochs = 0

    print("\n===== FULL TRAINING 97 LOP =====")

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== EPOCH {epoch}/{args.epochs} =====")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer,
            train_criterion, scaler, device, amp
        )

        val_loss, val_acc, val_f1 = validate(
            model, val_loader, val_criterion,
            device, amp, len(classes)
        )

        scheduler.step(val_f1)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1
        })

        print(
            f"\nEpoch {epoch}/{args.epochs} | "
            f"Train: {train_acc:.2%} | "
            f"Val: {val_acc:.2%} | "
            f"Macro F1: {val_f1:.4f}",
            flush=True
        )

        with (output / "history.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(history[0])
            )
            writer.writeheader()
            writer.writerows(history)

        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            bad_epochs = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": classes,
                "val_macro_f1": val_f1,
                "epoch": epoch,
                "train_images": len(train_rows)
            }, output / "best_plant_model.pth")

            print("=> Da luu checkpoint tot nhat")
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print("Early stopping!")
            break

    print("\n===== HOAN THANH =====")
    print("Best Validation Macro F1:", round(best_f1, 4))
    print("Ket qua:", output)
    print("Tap Test chua duoc su dung.")


if __name__ == "__main__":
    main()
