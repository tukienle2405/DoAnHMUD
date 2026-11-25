
# -*- coding: utf-8 -*-
"""
Nhan dien rieng 18 loai cay - MobileNetV2.

Su dung manifest 97 lop da co:
- Giu nguyen Train / Validation / Test
- Moi nhom anh gan trung chi lay 1 dai dien khi train
- Lay mau can bang theo cay va theo nhan benh
- Khoi tao backbone tu checkpoint 97 lop
- Khong sua hoac ghi de model 97 lop

Chay:
python train_18_crops.py --check_only
python -u train_18_crops.py --epochs 6 --batch_size 8
"""

import os
import csv
import json
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

ROOT = Path(r"D:\DoanHMvUD")

os.environ["TORCH_HOME"] = str(
    ROOT / ".cache" / "torch"
)

import numpy as np
import torch
import torch.nn as nn

from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from model_architecture import get_mobilenet_v2_model


MANIFEST = (
    ROOT / "dataset_review"
    / "dataset_97_lop"
    / "split_97_lop"
    / "dataset_manifest_SPLIT_DU_KIEN.csv"
)

DEFAULT_INITIAL = (
    ROOT / "training_runs"
    / "plant97_full_20261118_064241"
    / "best_plant_model.pth"
)

RUNS = ROOT / "training_runs"

AUDIT_DIR = (
    ROOT / "dataset_review" / "dataset_18_cay"
)

CROPS = sorted([
    "Banana", "Cashew", "Chili", "Coffee",
    "Corn", "Durian", "Guava", "Jackfruit",
    "Lemon", "Mango", "Papaya", "Pepper__bell",
    "Potato", "Rice", "Rubber", "Sugarcane",
    "Tea", "Tomato"
])

VALID_SPLITS = {"train", "val", "test"}


def crop_of(label):
    """Chuyen nhan benh ve ten cay."""

    if label.startswith("Pepper__bell"):
        return "Pepper__bell"

    if label.startswith("Tomato"):
        return "Tomato"

    if "___" not in label:
        raise ValueError("Nhan khong hop le: " + label)

    return label.split("___", 1)[0]


def source_group(row):
    """ID nhom anh trong pham vi tung lop."""

    if row["source"] == "TEA_GROUPED":
        group = row.get("group_id", "").strip()
    else:
        group = (
            row.get("near_group_id", "").strip()
            or row.get("group_id", "").strip()
        )

    return group or row["path"]


def read_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    with MANIFEST.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Manifest rong.")

    required = {
        "path", "class", "split",
        "source", "group_id", "near_group_id"
    }

    if not required.issubset(rows[0]):
        raise ValueError("Manifest thieu cot bat buoc.")

    seen_groups = {}
    seen_paths = set()

    valid_rows = []

    for row in rows:
        split = row["split"]

        if split not in VALID_SPLITS:
            continue

        label = row["class"]
        crop = crop_of(label)

        if crop not in CROPS:
            raise ValueError(
                f"Loai cay ngoai 18 lop: {crop}"
            )

        if row["path"] in seen_paths:
            raise ValueError(
                "Trung duong dan anh: " + row["path"]
            )

        seen_paths.add(row["path"])

        # Group ID chi co y nghia ben trong nhan.
        key = (label, source_group(row))

        previous = seen_groups.setdefault(key, split)

        if previous != split:
            raise ValueError(
                f"Ro ri nhom giua cac tap: {key}"
            )

        row["crop_18"] = crop
        valid_rows.append(row)

    actual_crops = {
        row["crop_18"] for row in valid_rows
    }

    if actual_crops != set(CROPS):
        raise ValueError(
            "Danh sach cay khong du 18 lop: "
            + str(sorted(actual_crops))
        )

    return valid_rows


def audit_data(rows):
    """Xuat bao cao ma khong huan luyen."""

    AUDIT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    records = []

    for crop in CROPS:
        selected = [
            r for r in rows
            if r["crop_18"] == crop
        ]

        counts = Counter(
            r["split"] for r in selected
        )

        train_groups = {
            (r["class"], source_group(r))
            for r in selected
            if r["split"] == "train"
        }

        review = sum(
            r.get(
                "needs_label_review", ""
            ).upper() == "YES"
            for r in selected
        )

        record = {
            "crop": crop,
            "train": counts["train"],
            "val": counts["val"],
            "test": counts["test"],
            "unique_train_groups": len(
                train_groups
            ),
            "needs_label_review": review
        }

        records.append(record)

    output = AUDIT_DIR / "phan_bo_18_cay.csv"

    with output.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )

        writer.writeheader()
        writer.writerows(records)

    print("\n===== KIEM TRA 18 LOAI CAY =====")

    for split in ("train", "val", "test"):
        count = sum(
            r["split"] == split for r in rows
        )
        print(f"{split}: {count}")

    print("So loai cay:", len(CROPS))

    print("\nPhan bo theo loai cay:")

    for item in records:
        print(
            f"{item['crop']:15s} "
            f"Train={item['train']:6d} "
            f"Val={item['val']:5d} "
            f"Test={item['test']:5d} "
            f"NhomTrain={item['unique_train_groups']:5d} "
            f"CanXemNhan={item['needs_label_review']:5d}"
        )

    print("\nDa xuat:", output)
    print(
        "Luu y: nhan can xac minh van co "
        "trong manifest. Chua dam bao "
        "khong ro ri theo cay/nguon anh."
    )


def choose_train_images(rows, maximum, seed):
    """
    Chi lay mot anh dai dien cua moi nhom.

    Can bang cac loai benh trong cung mot cay,
    de model nhan dien cay khong chi hoc
    bieu hien cua mot benh pho bien.
    """

    rng = random.Random(seed)

    grouped = defaultdict(list)

    for row in rows:
        if row["split"] != "train":
            continue

        group_key = (
            row["class"],
            source_group(row)
        )

        grouped[group_key].append(row)

    by_crop = defaultdict(
        lambda: defaultdict(list)
    )

    for (disease, group), items in grouped.items():
        representative = rng.choice(items)

        crop = representative["crop_18"]

        by_crop[crop][disease].append(
            representative
        )

    selected = []

    for crop in CROPS:
        buckets = by_crop[crop]

        for disease in buckets:
            rng.shuffle(buckets[disease])

        # 0 = lay tat ca nhom rieng biet.
        limit = (
            maximum if maximum > 0
            else sum(len(v) for v in buckets.values())
        )

        crop_selected = []

        # Lay luan phien giua cac nhan benh.
        # Khong lap lai anh, khong oversample.
        while len(crop_selected) < limit:
            added = False

            for disease in sorted(buckets):
                if not buckets[disease]:
                    continue

                crop_selected.append(
                    buckets[disease].pop()
                )

                added = True

                if len(crop_selected) >= limit:
                    break

            if not added:
                break

        selected.extend(crop_selected)

        print(
            f"{crop:15s}: "
            f"{len(crop_selected)} anh Train"
        )

    rng.shuffle(selected)

    return selected


class CropDataset(Dataset):
    def __init__(self, rows, crop_to_id, training=False):
        self.rows = rows
        self.crop_to_id = crop_to_id

        operations = [
            transforms.Resize((224, 224))
        ]

        if training:
            operations += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=12,
                    translate=(0.05, 0.05),
                    scale=(0.90, 1.05)
                ),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.20,
                        contrast=0.20,
                        saturation=0.15
                    )
                ], p=0.7)
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
            tensor = self.transform(img)

        return (
            tensor,
            self.crop_to_id[row["crop_18"]]
        )


def get_model(init_path, device):
    """
    Dung backbone 97 lop da hoc.
    Thay classifier thanh 18 lop.
    """

    if init_path.lower() == "none":
        model = get_mobilenet_v2_model(
            num_classes=18,
            pretrained=True
        )

        print("Khoi tao tu ImageNet.")

    else:
        path = Path(init_path)

        if not path.is_file():
            raise FileNotFoundError(path)

        old = torch.load(
            path, map_location="cpu",
            weights_only=True
        )

        model = get_mobilenet_v2_model(
            num_classes=18,
            pretrained=False
        )

        state = old["model_state_dict"]

        features = {
            key[len("features."):]: value
            for key, value in state.items()
            if key.startswith("features.")
        }

        model.features.load_state_dict(
            features, strict=True
        )

        print(
            "Khoi tao backbone 97 lop tu:",
            path
        )

    return model.to(device)


def evaluate(model, loader, device, amp):
    model.eval()

    actual = []
    predicted = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(
                device, non_blocking=True
            )

            with torch.autocast(
                device_type=device.type,
                enabled=amp
            ):
                logits = model(images)

            preds = logits.argmax(1)

            actual.extend(labels.tolist())
            predicted.extend(
                preds.cpu().tolist()
            )

    accuracy = accuracy_score(
        actual, predicted
    )

    macro_f1 = f1_score(
        actual, predicted,
        labels=list(range(18)),
        average="macro",
        zero_division=0
    )

    return (
        float(accuracy),
        float(macro_f1),
        actual,
        predicted
    )


def write_evaluation(
    folder, prefix, classes,
    accuracy, macro_f1,
    actual, predicted
):
    folder.mkdir(
        parents=True, exist_ok=True
    )

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "image_count": len(actual),
        "class_count": len(classes)
    }

    (folder / f"{prefix}_metrics.json").write_text(
        json.dumps(
            metrics, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0
    )

    (
        folder / f"{prefix}_classification_report.txt"
    ).write_text(
        report, encoding="utf-8"
    )

    cm = confusion_matrix(
        actual, predicted,
        labels=list(range(len(classes)))
    )

    with (
        folder / f"{prefix}_confusion_matrix.csv"
    ).open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["actual/predicted"] + classes
        )

        for name, values in zip(classes, cm):
            writer.writerow([
                name, *values.tolist()
            ])

    print("\n===== KET QUA =====")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)


def make_loader(
    rows, crop_to_id, batch_size,
    training=False, pin=False
):
    dataset = CropDataset(
        rows,
        crop_to_id,
        training=training
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=0,
        pin_memory=pin,
        drop_last=training
    )


def load_trained_checkpoint(path, device):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    classes = checkpoint["class_names"]

    if len(classes) != 18:
        raise ValueError(
            "Checkpoint khong phai 18 lop cay."
        )

    model = get_mobilenet_v2_model(
        num_classes=18,
        pretrained=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, classes


def predict_external(path, checkpoint, device):
    model, classes = load_trained_checkpoint(
        checkpoint, device
    )

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(
            img
        ).convert("RGB")

        transform = CropDataset(
            [], {}, training=False
        ).transform

        tensor = transform(
            img
        ).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)

        probs = torch.softmax(
            logits, dim=1
        )[0].cpu().numpy()

    top = np.argsort(probs)[::-1][:5]

    print("\n===== DU DOAN ANH NGOAI =====")
    print("Anh:", path)

    for i in top:
        print(
            f"{classes[i]:15s} "
            f"{probs[i] * 100:.2f}%"
        )

    print(
        "Chu y: Softmax chua duoc hieu chuan; "
        "khong phai xac suat dung thuc te."
    )


def train(args, rows, device, amp):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if amp:
        torch.cuda.manual_seed_all(args.seed)

    train_rows = choose_train_images(
        rows,
        args.max_train_per_crop,
        args.seed
    )

    val_rows = [
        r for r in rows
        if r["split"] == "val"
    ]

    mapping = {
        name: i for i, name in enumerate(CROPS)
    }

    if len(train_rows) < args.batch_size:
        raise ValueError(
            "Khong du anh cho batch_size."
        )

    counts = Counter(
        r["crop_18"] for r in train_rows
    )

    if any(counts[crop] == 0 for crop in CROPS):
        raise ValueError(
            "Co loai cay khong co anh Train."
        )

    train_loader = make_loader(
        train_rows, mapping,
        args.batch_size,
        training=True,
        pin=amp
    )

    val_loader = make_loader(
        val_rows, mapping,
        args.batch_size,
        pin=amp
    )

    model = get_model(
        args.init_checkpoint,
        device
    )

    median = sorted(
        counts.values()
    )[len(CROPS) // 2]

    weights = [
        min(
            3.0,
            max(
                0.5,
                math.sqrt(median / counts[crop])
            )
        )
        for crop in CROPS
    ]

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            weights,
            dtype=torch.float32,
            device=device
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p for p in model.features.parameters()
                    if p.requires_grad
                ],
                "lr": args.lr * 0.3
            },
            {
                "params": model.classifier.parameters(),
                "lr": args.lr
            }
        ],
        weight_decay=1e-4
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp
    )

    run_dir = RUNS / datetime.now().strftime(
        "crop18_%Y%m%d_%H%M%S"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    (run_dir / "training_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "manifest": str(MANIFEST),
                "train_images": len(train_rows),
                "val_images": len(val_rows),
                "device": str(device)
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    (run_dir / "class_names.json").write_text(
        json.dumps(
            CROPS,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    best_f1 = -1.0
    bad_epochs = 0
    history = []

    print("\n===== TRAIN 18 LOAI CAY =====")
    print("Train:", len(train_rows))
    print("Validation:", len(val_rows))
    print("Batch:", len(train_loader))
    print("Device:", device)
    print("Thu muc:", run_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total = 0
        correct = 0
        loss_sum = 0.0

        print(
            f"\n===== EPOCH {epoch}/{args.epochs} =====",
            flush=True
        )

        for step, (images, labels) in enumerate(
            train_loader, 1
        ):
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
                logits = model(images)
                loss = criterion(
                    logits, labels
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            scaler.step(optimizer)
            scaler.update()

            n = labels.size(0)

            total += n
            loss_sum += loss.item() * n
            correct += (
                logits.argmax(1) == labels
            ).sum().item()

            if (
                step % 100 == 0
                or step == len(train_loader)
            ):
                print(
                    f"Train: {step}/{len(train_loader)} "
                    f"({100 * step / len(train_loader):.1f}%)",
                    flush=True
                )

        val_acc, val_f1, actual, predicted = (
            evaluate(
                model, val_loader, device, amp
            )
        )

        train_acc = correct / total
        train_loss = loss_sum / total

        print(
            f"Epoch {epoch} | "
            f"Train: {train_acc:.2%} | "
            f"Val: {val_acc:.2%} | "
            f"Macro F1: {val_f1:.4f}",
            flush=True
        )

        history.append({
            "epoch": epoch,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        })

        with (run_dir / "history.csv").open(
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(history[0])
            )

            writer.writeheader()
            writer.writerows(history)

        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            bad_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CROPS,
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "task": "crop18"
                },
                run_dir / "best_crop18_model.pth"
            )

            write_evaluation(
                run_dir, "validation",
                CROPS, val_acc, val_f1,
                actual, predicted
            )

            print("=> Da luu model 18 cay tot nhat.")

        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print("Early stopping.")
            break

    print("\n===== HOAN THANH =====")
    print("Best Validation Macro F1:", best_f1)
    print(
        "Checkpoint:",
        run_dir / "best_crop18_model.pth"
    )
    print("Chua danh gia tren Test.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check_only", action="store_true"
    )

    parser.add_argument(
        "--epochs", type=int, default=6
    )

    parser.add_argument(
        "--batch_size", type=int, default=8
    )

    parser.add_argument(
        "--max_train_per_crop",
        type=int,
        default=600,
        help="0 = moi nhom Train lay 1 anh"
    )

    parser.add_argument(
        "--lr", type=float, default=0.0002
    )

    parser.add_argument(
        "--patience", type=int, default=3
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--init_checkpoint",
        default=str(DEFAULT_INITIAL)
    )

    parser.add_argument(
        "--test_checkpoint",
        type=str,
        default=""
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=""
    )

    parser.add_argument(
        "--predict_image",
        type=str,
        default=""
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    amp = device.type == "cuda"

    if args.predict_image:
        if not args.checkpoint:
            parser.error(
                "--predict_image can --checkpoint"
            )

        predict_external(
            args.predict_image,
            args.checkpoint,
            device
        )
        return

    rows = read_manifest()
    audit_data(rows)

    if args.check_only:
        print(
            "\nKIEM TRA XONG. CHUA HUAN LUYEN."
        )
        return

    if args.test_checkpoint:
        model, classes = load_trained_checkpoint(
            args.test_checkpoint,
            device
        )

        mapping = {
            name: i for i, name in enumerate(classes)
        }

        test_rows = [
            r for r in rows
            if r["split"] == "test"
        ]

        loader = make_loader(
            test_rows,
            mapping,
            args.batch_size,
            pin=amp
        )

        acc, f1, actual, predicted = evaluate(
            model, loader, device, amp
        )

        write_evaluation(
            Path(args.test_checkpoint).parent,
            "test_crop18",
            classes, acc, f1,
            actual, predicted
        )

        return

    train(args, rows, device, amp)


if __name__ == "__main__":
    main()
