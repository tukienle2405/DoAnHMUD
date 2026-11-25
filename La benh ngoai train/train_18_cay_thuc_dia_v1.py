
# -*- coding: utf-8 -*-
"""
PILOT: Cai thien nhan dien 18 loai cay tren anh ngoai Train.

- Khong thay doi dataset goc
- Khong ghi de checkpoint cu
- Ket hop anh Train cu va PlantDoc train
- Dung PlantDoc val de chon checkpoint
- Duy tri ket qua Validation cua du 18 cay
- Test thuc dia chi chay sau khi chon checkpoint
- CUDA, mixed precision, ETA
"""

import sys
import csv
import json
import time
import random
import argparse
import importlib.util
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import f1_score

ROOT = Path(r"D:\DoanHMvUD")
sys.path.insert(0, str(ROOT))

FIELD_CSV = (
    ROOT / "external_field_pool_v2"
    / "field_adaptation_v1"
    / "EXTERNAL_SPLIT_LOCKED.csv"
)

OLD_CHECKPOINT = (
    ROOT / "training_runs"
    / "crop18_final_20261121_171921"
    / "best_crop18_model.pth"
)

EXPECTED_FIELD_CROPS = (
    "Corn", "Pepper__bell", "Potato", "Tomato"
)

# Tim file train goc
OLD_SCRIPT_LOCATIONS = [
    ROOT / "train_18_crops_final.py",
    ROOT / "La benh ngoai train" / "train_18_crops_final.py",
]

old_script = next(
    (p for p in OLD_SCRIPT_LOCATIONS if p.is_file()),
    None
)

if old_script is None:
    raise FileNotFoundError(
        "Khong tim thay train_18_crops_final.py "
        "tai thu muc goc hoac 'La benh ngoai train'."
    )

spec = importlib.util.spec_from_file_location(
    "original_crop18", old_script
)

old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

CROPS = old.CROPS
CROP_TO_ID = {crop: i for i, crop in enumerate(CROPS)}


def duration(seconds):
    seconds = max(0, int(seconds))
    h, remain = divmod(seconds, 3600)
    m, s = divmod(remain, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_field_rows():
    if not FIELD_CSV.is_file():
        raise FileNotFoundError(FIELD_CSV)

    with FIELD_CSV.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        records = list(csv.DictReader(f))

    clusters = defaultdict(set)
    groups = defaultdict(set)

    for row in records:
        clusters[row["cluster"]].add(row["split"])
        groups[row["cluster"]].add(row["crop"])

    if any(len(v) > 1 for v in clusters.values()):
        raise ValueError(
            "PHAT HIEN NHOM ANH NAM TRONG NHIEU SPLIT!"
        )

    train, val, test = [], [], []

    for row in records:
        if row["split"] == "exclude":
            continue

        if row["reason"] != "OK":
            continue

        path = Path(row["path"])

        if not path.is_file():
            raise FileNotFoundError(path)

        if row["crop"] not in EXPECTED_FIELD_CROPS:
            raise ValueError(
                f"Nhan cay khong hop le: {row['crop']}"
            )

        record = {
            "path": str(path),
            "crop": row["crop"],
            "class": row["crop"],
            "group": row["group"],
        }

        if row["split"] == "train":
            train.append(record)
        elif row["split"] == "val":
            val.append(record)
        elif row["split"] == "test":
            test.append(record)

    for name, rows in [
        ("train", train),
        ("val", val),
        ("test", test)
    ]:
        found = {r["crop"] for r in rows}

        if found != set(EXPECTED_FIELD_CROPS):
            raise ValueError(
                f"{name} thieu cay: "
                f"{set(EXPECTED_FIELD_CROPS) - found}"
            )

    return train, val, test


def stratified_sample(rows, max_per_crop, seed):
    rng = random.Random(seed)
    buckets = defaultdict(list)

    for row in rows:
        buckets[row["crop"]].append(row)

    selected = []

    for crop in sorted(buckets):
        items = buckets[crop]
        selected.extend(
            rng.sample(
                items,
                min(len(items), max_per_crop)
            )
        )

    return selected


class WeightedDataset(Dataset):
    def __init__(self, rows, weight):
        self.dataset = old.LeafDataset(rows, train=True)
        self.weight = float(weight)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]

        return image, target, self.weight


def make_eval_loader(rows, batch_size, device):
    return DataLoader(
        old.LeafDataset(rows, train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


@torch.inference_mode()
def evaluate(model, rows, device, batch_size):
    loader = make_eval_loader(
        rows, batch_size, device
    )

    actual = []
    predicted = []

    model.eval()

    for images, targets in loader:
        images = images.to(device, non_blocking=True)

        logits = model(images)
        predictions = logits.argmax(1).cpu()

        actual.extend(targets.tolist())
        predicted.extend(predictions.tolist())

    if not actual:
        raise ValueError("Tap danh gia rong.")

    accuracy = sum(
        a == b for a, b in zip(actual, predicted)
    ) / len(actual)

    macro18 = f1_score(
        actual,
        predicted,
        labels=list(range(18)),
        average="macro",
        zero_division=0
    )

    stats = {}

    for crop in EXPECTED_FIELD_CROPS:
        class_id = CROP_TO_ID[crop]

        positions = [
            i for i, a in enumerate(actual)
            if a == class_id
        ]

        if positions:
            correct = sum(
                predicted[i] == class_id
                for i in positions
            )

            stats[crop] = {
                "correct": correct,
                "total": len(positions),
                "recall": correct / len(positions)
            }

    macro4 = (
        sum(x["recall"] for x in stats.values())
        / len(stats)
    ) if stats else 0.0

    return {
        "accuracy": accuracy,
        "macro18": float(macro18),
        "macro4": macro4,
        "stats": stats,
        "count": len(actual)
    }


def create_model(device):
    checkpoint = torch.load(
        OLD_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    if list(checkpoint["class_names"]) != CROPS:
        raise ValueError(
            "Thu tu 18 nhan khong khop checkpoint!"
        )

    model = old.get_mobilenet_v2_model(
        num_classes=18,
        pretrained=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    # Giu phan lon backbone, fine-tune 8 block cuoi
    model.features.requires_grad_(False)

    for layer in list(
        model.features.children()
    )[-8:]:
        layer.requires_grad_(True)

    model.classifier.requires_grad_(True)

    return model.to(device), checkpoint


def save_csv(path, records):
    if not records:
        return

    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=2
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--old_per_crop", type=int, default=80
    )
    parser.add_argument(
        "--field_per_crop", type=int, default=100
    )
    parser.add_argument(
        "--field_weight", type=float, default=3.0
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("epochs phai >= 1")

    if args.batch_size < 2:
        raise ValueError("batch_size phai >= 2")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    print("THIET BI:", device)
    print("BAT DAU:", datetime.now())

    session_start = time.perf_counter()

    # Doc du lieu goc theo manifest cu
    original_rows = old.load_rows()

    original_val = [
        r for r in original_rows
        if r["split"] == "val"
        and r.get("needs_label_review", "NO") == "NO"
    ]

    original_val = stratified_sample(
        original_val, 50, args.seed
    )

    # Doc du lieu thuc dia da khoa
    field_train, field_val, field_test = (
        load_field_rows()
    )

    print("\n===== DU LIEU =====")
    print("PlantDoc Train:", len(field_train))
    print("PlantDoc Val:", len(field_val))
    print("PlantDoc Test:", len(field_test))
    print("Old Val mau:", len(original_val))

    # Test KHONG duoc dung trong vong train
    model, original_checkpoint = create_model(device)

    print("\n===== BASELINE VALIDATION =====")

    base_field = evaluate(
        model, field_val, device, args.batch_size
    )

    base_old = evaluate(
        model, original_val, device, args.batch_size
    )

    print(
        "Field Val Accuracy:",
        f"{base_field['accuracy']:.2%}"
    )
    print(
        "Field Val Macro Recall 4 cay:",
        f"{base_field['macro4']:.2%}"
    )
    print(
        "Old Val Macro F1 18 cay:",
        f"{base_old['macro18']:.4f}"
    )

    print("\nRecall tung cay ban dau:")

    for crop, info in base_field["stats"].items():
        print(
            crop,
            f"{info['correct']}/{info['total']}",
            f"= {info['recall']:.2%}"
        )

    # Thu muc rieng, khong ghi de model cu
    run_dir = (
        ROOT / "training_runs"
        / datetime.now().strftime(
            "crop18_field_pilot_%Y%m%d_%H%M%S"
        )
    )

    run_dir.mkdir(parents=True, exist_ok=False)

    save_path = run_dir / "best_field_candidate.pth"

    (run_dir / "config.json").write_text(
        json.dumps(
            vars(args),
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # Lay mau Train cu: du 18 cay
    cycle = old.BalancedCycle(
        original_rows, args.seed
    )

    feature_params = [
        p for p in model.features.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": feature_params,
                "lr": 0.00001
            },
            {
                "params": model.classifier.parameters(),
                "lr": 0.0001
            }
        ],
        weight_decay=0.0001
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    )

    amp = device.type == "cuda"

    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp
    )

    history = []
    best_score = base_field["macro4"] + 0.03
    accepted = False

    field_by_crop = defaultdict(list)

    for row in field_train:
        field_by_crop[row["crop"]].append(row)

    for epoch in range(1, args.epochs + 1):

        # Train goc - can bang 18 cay
        old_train = cycle.sample(
            args.old_per_crop, epoch
        )

        # Anh thuc dia: lay mau can bang theo cay
        rng = random.Random(
            args.seed + epoch * 101
        )

        selected_field = []

        for crop in EXPECTED_FIELD_CROPS:
            items = field_by_crop[crop]

            selected_field.extend(
                rng.sample(
                    items,
                    min(len(items), args.field_per_crop)
                )
            )

        dataset = ConcatDataset([
            WeightedDataset(old_train, 1.0),
            WeightedDataset(
                selected_field, args.field_weight
            )
        ])

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=amp
        )

        print(
            f"\n===== EPOCH {epoch}/{args.epochs} ====="
        )

        print("Old Train:", len(old_train))
        print("Field Train:", len(selected_field))

        model.train()

        # BatchNorm backbone giu thong ke on dinh
        old.freeze_feature_bn(model)

        train_start = time.perf_counter()
        total_loss = 0.0

        for step, (images, targets, weights) in enumerate(
            loader, 1
        ):
            images = images.to(
                device, non_blocking=True
            )
            targets = targets.to(
                device, non_blocking=True
            )
            weights = weights.to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                enabled=amp
            ):
                logits = model(images)

                losses = F.cross_entropy(
                    logits,
                    targets,
                    reduction="none"
                )

                loss = (
                    (losses * weights).sum()
                    / weights.sum()
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())

            if step % 40 == 0 or step == len(loader):
                elapsed = (
                    time.perf_counter() - train_start
                )

                speed = elapsed / step

                left = (
                    len(loader) - step
                ) * speed

                print(
                    f"Train {step}/{len(loader)} "
                    f"({step / len(loader):.1%}) | "
                    f"Da chay {duration(elapsed)} | "
                    f"Con lai epoch ~{duration(left)} | "
                    f"Du kien xong train epoch "
                    f"{datetime.now() + timedelta(seconds=left):%H:%M:%S}",
                    flush=True
                )

        print("\nDang danh gia Validation...")

        field_result = evaluate(
            model, field_val,
            device, args.batch_size
        )

        old_result = evaluate(
            model, original_val,
            device, args.batch_size
        )

        # Dieu kien giu model nghiem ngat
        # 1. Macro recall tren anh field tang >= 3 diem %
        # 2. Tomato, Potato, Pepper deu cai thien
        # 3. Corn khong giam qua 15 diem %
        # 4. Macro F1 18 cay cu khong giam qua 4 diem %

        improves_hard_crops = all(
            field_result["stats"][crop]["recall"]
            > base_field["stats"][crop]["recall"]
            for crop in (
                "Tomato", "Potato", "Pepper__bell"
            )
        )

        corn_preserved = (
            field_result["stats"]["Corn"]["recall"]
            >= base_field["stats"]["Corn"]["recall"] - 0.15
        )

        old_preserved = (
            old_result["macro18"]
            >= base_old["macro18"] - 0.04
        )

        new_macro = field_result["macro4"]

        qualifies = (
            new_macro > best_score
            and improves_hard_crops
            and corn_preserved
            and old_preserved
        )

        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(loader)),
            "field_accuracy": field_result["accuracy"],
            "field_macro_recall4": new_macro,
            "old_macro_f1_18": old_result["macro18"],
            "improves_3_hard_crops": improves_hard_crops,
            "corn_preserved": corn_preserved,
            "old_preserved": old_preserved,
            "qualified": qualifies
        }

        history.append(record)

        save_csv(run_dir / "history.csv", history)

        print(
            f"Epoch {epoch} | "
            f"Field Accuracy {field_result['accuracy']:.2%} | "
            f"Field Macro Recall {new_macro:.2%} | "
            f"Old Macro F1 {old_result['macro18']:.4f}"
        )

        for crop, info in field_result["stats"].items():
            print(
                f"  {crop:15s} "
                f"{info['correct']:3d}/{info['total']:3d} "
                f"{info['recall']:.2%}"
            )

        if qualifies:
            best_score = new_macro
            accepted = True

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CROPS,
                    "field_val_macro_recall": new_macro,
                    "old_val_macro_f1": old_result["macro18"],
                    "epoch": epoch
                },
                save_path
            )

            print(
                "=> DA LUU CHECKPOINT UNG VIEN:",
                save_path
            )
        else:
            print(
                "=> CHUA DAT DIEU KIEN. "
                "KHONG LUU CHECKPOINT."
            )

        scheduler.step()

    # Chi mo Test sau khi da chon checkpoint
    if accepted:

        print("\n===== KIEM TRA TEST NGOAI TRAIN =====")

        best = torch.load(
            save_path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(
            best["model_state_dict"]
        )

        candidate_test = evaluate(
            model, field_test,
            device, args.batch_size
        )

        # So sanh voi checkpoint goc tren cung Test
        model.load_state_dict(
            original_checkpoint["model_state_dict"]
        )

        original_test = evaluate(
            model, field_test,
            device, args.batch_size
        )

        comparisons = []

        for crop in EXPECTED_FIELD_CROPS:
            before = original_test["stats"][crop]
            after = candidate_test["stats"][crop]

            comparisons.append({
                "crop": crop,
                "test_images": before["total"],
                "old_correct": before["correct"],
                "new_correct": after["correct"],
                "old_recall": before["recall"],
                "new_recall": after["recall"]
            })

        save_csv(
            run_dir / "SO_SANH_TEST_THUC_DIA.csv",
            comparisons
        )

        improved = (
            candidate_test["accuracy"]
            > original_test["accuracy"]
            and candidate_test["macro4"]
            > original_test["macro4"]
        )

        summary = {
            "original_test_accuracy":
                original_test["accuracy"],
            "candidate_test_accuracy":
                candidate_test["accuracy"],
            "original_test_macro_recall4":
                original_test["macro4"],
            "candidate_test_macro_recall4":
                candidate_test["macro4"],
            "improved_on_test": improved,
            "note":
                "Chi la test 4 cay PlantDoc, "
                "chua du 18 cay ngoai thuc te."
        }

        (run_dir / "test_summary.json").write_text(
            json.dumps(
                summary,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        print(
            "Test Accuracy cu:",
            f"{original_test['accuracy']:.2%}"
        )
        print(
            "Test Accuracy moi:",
            f"{candidate_test['accuracy']:.2%}"
        )

        print(
            "Test Macro Recall cu:",
            f"{original_test['macro4']:.2%}"
        )
        print(
            "Test Macro Recall moi:",
            f"{candidate_test['macro4']:.2%}"
        )

        if improved:
            print(
                "KET QUA: UNG VIEN CO CAI THIEN "
                "TREN PLANTDOC TEST."
            )
        else:
            print(
                "KET QUA: KHONG DAT. "
                "KHONG NEN DOI MODEL DANG DUNG."
            )

    else:
        print(
            "\nKHONG CO CHECKPOINT MOI NAO "
            "DAT DIEU KIEN VALIDATION."
        )
        print(
            "GIU NGUYEN MODEL 18 CAY CU. "
            "KHONG MO TAP TEST."
        )

    print("\n===== HOAN THANH =====")
    print("Thoi gian:", duration(
        time.perf_counter() - session_start
    ))
    print("Hoan thanh luc:", datetime.now())
    print("Bao cao:", run_dir)


if __name__ == "__main__":
    main()
