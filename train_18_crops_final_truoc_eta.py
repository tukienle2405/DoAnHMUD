
# -*- coding: utf-8 -*-
"""
FINAL - Nhan dien 18 loai cay.

- Backbone tu model 97 lop
- Can bang cay va cac nhan benh
- Xoay vong nhom anh qua moi epoch
- Mixed precision tren NVIDIA GPU
- Early stopping
- Luu checkpoint moi epoch de co the resume
- Tu danh gia Test sau khi huan luyen
- Du doan anh ngoai dataset neu co

Khong sua model 97 lop va anh goc.
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
os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

from model_architecture import get_mobilenet_v2_model

MANIFEST = (
    ROOT / "dataset_review" / "dataset_97_lop"
    / "split_97_lop"
    / "dataset_manifest_SPLIT_DU_KIEN.csv"
)

MODEL97 = (
    ROOT / "training_runs"
    / "plant97_full_20261118_064241"
    / "best_plant_model.pth"
)

CROPS = sorted([
    "Banana", "Cashew", "Chili", "Coffee",
    "Corn", "Durian", "Guava", "Jackfruit",
    "Lemon", "Mango", "Papaya", "Pepper__bell",
    "Potato", "Rice", "Rubber", "Sugarcane",
    "Tea", "Tomato"
])

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def crop_of(label):
    if label.startswith("Pepper__bell"):
        return "Pepper__bell"
    if label.startswith("Tomato"):
        return "Tomato"
    return label.split("___", 1)[0]


def group_of(row):
    if row["source"] == "TEA_GROUPED":
        return row["group_id"]

    return (
        row.get("near_group_id", "").strip()
        or row.get("group_id", "").strip()
        or row["path"]
    )


def load_rows():
    with MANIFEST.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    valid = []
    groups = {}
    seen_paths = set()

    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            continue

        crop = crop_of(row["class"])

        if crop not in CROPS:
            raise ValueError(
                f"Loai cay khong hop le: {crop}"
            )

        if row["path"] in seen_paths:
            raise ValueError(
                "Trung duong dan: " + row["path"]
            )

        seen_paths.add(row["path"])

        key = (
            row["class"],
            row["source"],
            group_of(row)
        )

        previous = groups.setdefault(
            key, row["split"]
        )

        if previous != row["split"]:
            raise ValueError(
                f"Ro ri nhom anh: {key}"
            )

        row["crop"] = crop
        valid.append(row)

    if {r["crop"] for r in valid} != set(CROPS):
        raise ValueError("Khong du 18 loai cay.")

    return valid


class BalancedCycle:
    """
    Moi nhom anh chi co mot anh dai dien.
    Cac epoch xoay vong qua toan bo nhom.
    Khong lap lai mot anh trong cung epoch.
    """

    def __init__(self, rows, seed):
        representatives = {}

        for row in rows:
            if row["split"] != "train":
                continue

            key = (
                row["class"],
                row["source"],
                group_of(row)
            )

            representatives.setdefault(key, row)

        self.buckets = defaultdict(
            lambda: defaultdict(list)
        )

        for row in representatives.values():
            self.buckets[row["crop"]][
                row["class"]
            ].append(row)

        self.positions = {}

        for crop in CROPS:
            for disease, items in self.buckets[crop].items():
                random.Random(
                    f"{seed}:{disease}"
                ).shuffle(items)

                self.positions[(crop, disease)] = 0

    def sample(self, per_crop, epoch):
        chosen = []

        for crop in CROPS:
            buckets = self.buckets[crop]

            total = sum(
                len(v) for v in buckets.values()
            )

            quota = (
                total if per_crop == 0
                else min(per_crop, total)
            )

            used = {
                name: 0 for name in buckets
            }

            current = []

            while len(current) < quota:
                added = False

                for disease in sorted(buckets):
                    items = buckets[disease]

                    if used[disease] >= len(items):
                        continue

                    position = (
                        self.positions[(crop, disease)]
                        + used[disease]
                    ) % len(items)

                    current.append(
                        items[position]
                    )

                    used[disease] += 1
                    added = True

                    if len(current) == quota:
                        break

                if not added:
                    break

            for disease in buckets:
                key = (crop, disease)

                self.positions[key] = (
                    self.positions[key]
                    + used[disease]
                ) % len(buckets[disease])

            chosen.extend(current)

        random.Random(42 + epoch).shuffle(chosen)
        return chosen

    def summary(self):
        return {
            crop: sum(
                len(v) for v in self.buckets[crop].values()
            )
            for crop in CROPS
        }


class LeafDataset(Dataset):
    def __init__(self, rows, train=False):
        self.rows = rows
        self.mapping = {
            crop: i for i, crop in enumerate(CROPS)
        }

        ops = []

        if train:
            ops += [
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.78, 1.0),
                    ratio=(0.85, 1.15)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.18,
                        contrast=0.18,
                        saturation=0.15
                    )
                ], p=0.75)
            ]
        else:
            ops += [
                transforms.Resize((224, 224))
            ]

        ops += [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]

        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["path"]) as img:
            img = ImageOps.exif_transpose(
                img
            ).convert("RGB")

            image = self.transform(img)

        return (
            image,
            self.mapping[row["crop"]]
        )


def make_loader(rows, batch, train, device):
    return DataLoader(
        LeafDataset(rows, train),
        batch_size=batch,
        shuffle=train,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=train
    )


def build_model(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    old = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model = get_mobilenet_v2_model(
        num_classes=18,
        pretrained=False
    )

    features = {
        key: value
        for key, value in old["model_state_dict"].items()
        if key.startswith("features.")
    }

    result = model.load_state_dict(
        features, strict=False
    )

    if result.unexpected_keys:
        raise ValueError(
            "Backbone khong khop checkpoint 97."
        )

    # Chi fine-tune 4 block sau cung.
    model.features.requires_grad_(False)

    for layer in list(
        model.features.children()
    )[-4:]:
        layer.requires_grad_(True)

    model.classifier.requires_grad_(True)

    return model.to(device)


def freeze_feature_bn(model):
    """
    Giu thong ke BatchNorm cua backbone,
    nhung van cho phep cap nhat convolution.
    """
    for layer in model.features.modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.eval()


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()

    actual = []
    predicted = []

    for images, labels in loader:
        images = images.to(
            device, non_blocking=True
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp
        ):
            logits = model(images)

        actual.extend(labels.tolist())
        predicted.extend(
            logits.argmax(1).cpu().tolist()
        )

    acc = accuracy_score(
        actual, predicted
    )

    f1 = f1_score(
        actual, predicted,
        labels=list(range(18)),
        average="macro",
        zero_division=0
    )

    return float(acc), float(f1), actual, predicted


def save_report(folder, name, results):
    acc, f1, actual, predicted = results

    report = classification_report(
        actual, predicted,
        labels=list(range(18)),
        target_names=CROPS,
        zero_division=0
    )

    (folder / f"{name}_report.txt").write_text(
        report, encoding="utf-8"
    )

    matrix = confusion_matrix(
        actual, predicted,
        labels=list(range(18))
    )

    with (folder / f"{name}_confusion.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            ["actual/predicted"] + CROPS
        )

        for crop, values in zip(CROPS, matrix):
            writer.writerow([
                crop, *values.tolist()
            ])

    (folder / f"{name}_metrics.json").write_text(
        json.dumps({
            "accuracy": acc,
            "macro_f1": f1,
            "images": len(actual)
        }, indent=2),
        encoding="utf-8"
    )

    print(
        f"{name}: Accuracy={acc:.2%} "
        f"Macro F1={f1:.4f}",
        flush=True
    )


def evaluate_external(model, run_dir, device):
    folder = ROOT / "external_test"

    if not folder.is_dir():
        print(
            "Chua co external_test. "
            "Bo qua kiem thu anh ngoai."
        )
        return

    paths = sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTS
    )

    if not paths:
        return

    transform = LeafDataset([], False).transform
    records = []

    model.eval()

    for path in paths:
        try:
            with Image.open(path) as img:
                image = ImageOps.exif_transpose(
                    img
                ).convert("RGB")

                tensor = transform(
                    image
                ).unsqueeze(0).to(device)

            with torch.inference_mode():
                logits = model(tensor)
                probs = torch.softmax(
                    logits, dim=1
                )[0].cpu().numpy()

            indices = np.argsort(
                probs
            )[::-1][:3]

            parts = path.relative_to(folder).parts

            # Ten thu muc con la nhan that.
            true_crop = (
                parts[0]
                if len(parts) >= 2
                and parts[0] in CROPS
                else ""
            )

            records.append({
                "path": str(path),
                "true_crop": true_crop,
                "predicted_crop": CROPS[indices[0]],
                "softmax": (
                    f"{float(probs[indices[0]]):.6f}"
                ),
                "correct": (
                    int(CROPS[indices[0]] == true_crop)
                    if true_crop else ""
                ),
                "top3": " | ".join(
                    CROPS[i] for i in indices
                )
            })

        except Exception as exc:
            print("Anh loi:", path, exc)

    if not records:
        return

    output = run_dir / "external_predictions.csv"

    with output.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )

        writer.writeheader()
        writer.writerows(records)

    labeled = [
        r for r in records
        if r["correct"] != ""
    ]

    print("\n===== ANH NGOAI DATASET =====")

    if labeled:
        correct = sum(
            r["correct"] for r in labeled
        )

        print(
            "Crop Accuracy:",
            f"{correct / len(labeled):.2%}"
        )

    print("Bao cao:", output)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--check_only", action="store_true"
    )
    parser.add_argument(
        "--epochs", type=int, default=8
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--per_crop_per_epoch",
        type=int, default=1200,
        help="0 = tat ca nhom anh moi epoch"
    )
    parser.add_argument(
        "--patience", type=int, default=3
    )
    parser.add_argument(
        "--resume", type=str, default=""
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    amp = device.type == "cuda"

    if amp:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    rows = load_rows()

    cycle = BalancedCycle(rows, args.seed)

    summary = cycle.summary()

    print("\n===== NHOM TRAIN THEO CAY =====")

    for crop in CROPS:
        print(
            f"{crop:15s}: "
            f"{summary[crop]:6d} nhom"
        )

    print("Thiet bi:", device)

    val_rows = [
        r for r in rows if r["split"] == "val"
    ]
    test_rows = [
        r for r in rows if r["split"] == "test"
    ]

    print("Validation:", len(val_rows))
    print("Test:", len(test_rows))

    if args.check_only:
        print(
            "KIEM TRA XONG. "
            "CHUA HUAN LUYEN."
        )
        return

    if args.batch_size < 2:
        raise ValueError(
            "Batch size phai >= 2 vi co BatchNorm1d."
        )

    model = build_model(
        MODEL97, device
    )

    head_params = list(
        model.classifier.parameters()
    )

    feature_params = [
        p for p in model.features.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": feature_params,
                "lr": 0.00002
            },
            {
                "params": head_params,
                "lr": 0.0002
            }
        ],
        weight_decay=0.0001
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    )

    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp
    )

    val_loader = make_loader(
        val_rows, 16, False, device
    )

    if args.resume:
        resume_path = Path(args.resume)

        state = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            state["model"]
        )

        optimizer.load_state_dict(
            state["optimizer"]
        )

        scheduler.load_state_dict(
            state["scheduler"]
        )

        scaler.load_state_dict(
            state["scaler"]
        )

        start_epoch = state["epoch"] + 1
        best_f1 = state["best_f1"]
        bad_epochs = state["bad_epochs"]
        history = state["history"]

        run_dir = resume_path.parent

        for old_epoch in range(1, start_epoch):
            cycle.sample(
                args.per_crop_per_epoch,
                old_epoch
            )

    else:
        run_dir = (
            ROOT / "training_runs"
            / datetime.now().strftime(
                "crop18_final_%Y%m%d_%H%M%S"
            )
        )

        run_dir.mkdir(
            parents=True, exist_ok=False
        )

        start_epoch = 1
        best_f1 = -1.0
        bad_epochs = 0
        history = []

        (run_dir / "config.json").write_text(
            json.dumps(
                vars(args),
                indent=2
            ),
            encoding="utf-8"
        )

    criterion_val = nn.CrossEntropyLoss()

    for epoch in range(
        start_epoch, args.epochs + 1
    ):
        train_rows = cycle.sample(
            args.per_crop_per_epoch,
            epoch
        )

        counts = Counter(
            r["crop"] for r in train_rows
        )

        median = sorted(
            counts.values()
        )[len(CROPS) // 2]

        weights = [
            min(
                3.0,
                max(
                    0.5,
                    math.sqrt(
                        median / counts[crop]
                    )
                )
            )
            for crop in CROPS
        ]

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                weights,
                dtype=torch.float32,
                device=device
            ),
            label_smoothing=0.04
        )

        loader = make_loader(
            train_rows,
            args.batch_size,
            True,
            device
        )

        model.train()
        freeze_feature_bn(model)

        total = 0
        correct = 0
        total_loss = 0.0

        print(
            f"\n===== EPOCH {epoch}/{args.epochs} ====="
        )
        print(
            "Anh Train epoch nay:",
            len(train_rows)
        )

        for step, (images, labels) in enumerate(
            loader, 1
        ):
            images = images.to(
                device, non_blocking=True
            )
            labels = labels.to(
                device, non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type=device.type,
                enabled=amp
            ):
                outputs = model(images)
                loss = criterion(
                    outputs, labels
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
            total_loss += loss.item() * n

            correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            if (
                step % 200 == 0
                or step == len(loader)
            ):
                print(
                    f"Train {step}/{len(loader)} "
                    f"({step / len(loader):.1%})",
                    flush=True
                )

        val_result = evaluate(
            model, val_loader, device, amp
        )

        val_acc, val_f1 = val_result[:2]

        print(
            f"Epoch {epoch} | "
            f"Train {correct / total:.2%} | "
            f"Val {val_acc:.2%} | "
            f"Macro F1 {val_f1:.4f}",
            flush=True
        )

        history.append({
            "epoch": epoch,
            "train_accuracy": correct / total,
            "train_loss": total_loss / total,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        })

        with (run_dir / "history.csv").open(
            "w", newline="",
            encoding="utf-8-sig"
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=list(history[0])
            )
            writer.writeheader()
            writer.writerows(history)

        if val_f1 > best_f1 + 0.0001:
            best_f1 = val_f1
            bad_epochs = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CROPS,
                "val_macro_f1": best_f1,
                "epoch": epoch
            }, run_dir / "best_crop18_model.pth")

            save_report(
                run_dir, "validation",
                val_result
            )

            print("=> Da luu model tot nhat.")

        else:
            bad_epochs += 1

        scheduler.step()

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "bad_epochs": bad_epochs,
            "history": history
        }, run_dir / "last_train_state.pth")

        if bad_epochs >= args.patience:
            print("Early stopping.")
            break

    best_path = (
        run_dir / "best_crop18_model.pth"
    )

    best = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        best["model_state_dict"]
    )

    print("\n===== DANH GIA TEST =====")

    test_loader = make_loader(
        test_rows, 16, False, device
    )

    test_result = evaluate(
        model, test_loader, device, amp
    )

    save_report(
        run_dir, "test",
        test_result
    )

    evaluate_external(
        model, run_dir, device
    )

    print("\n===== HOAN THANH =====")
    print("Best Validation Macro F1:", best_f1)
    print("Model:", best_path)
    print("Bao cao:", run_dir)


if __name__ == "__main__":
    main()
