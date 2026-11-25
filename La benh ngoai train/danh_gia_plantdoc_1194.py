
from pathlib import Path
from collections import defaultdict
import sys
import csv
import argparse

import torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(r"D:\DoanHMvUD")
sys.path.insert(0, str(ROOT))

from model_architecture import get_mobilenet_v2_model

DATA_DIR = (
    ROOT / "external_field_pool_v2"
    / "PlantDoc_khong_trung_SHA256"
)

MODEL_18 = (
    ROOT / "training_runs"
    / "crop18_final_20261121_171921"
    / "best_crop18_model.pth"
)

MODEL_97 = (
    ROOT / "training_runs"
    / "plant97_full_20261118_064241"
    / "best_plant_model.pth"
)

OUTPUT = ROOT / "external_field_pool_v2" / "danh_gia_plantdoc"

# Ánh xạ nhãn PlantDoc sang lớp của mô hình.
# None = nhãn nguồn chưa đủ cụ thể để chấm đúng bệnh.
LABEL_MAP = {
    "Bell_pepper leaf":
        ("Pepper__bell", "Pepper__bell___healthy"),
    "Bell_pepper leaf spot":
        ("Pepper__bell", None),

    "Corn Gray leaf spot":
        ("Corn", "Corn___Gray_leaf_spot"),
    "Corn leaf blight":
        ("Corn", "Corn___Blight"),
    "Corn rust leaf":
        ("Corn", "Corn___Common_rust"),

    "Potato leaf early blight":
        ("Potato", "Potato___Early_blight"),
    "Potato leaf late blight":
        ("Potato", "Potato___Late_blight"),

    "Tomato Early blight leaf":
        ("Tomato", "Tomato_Early_blight"),
    "Tomato leaf":
        ("Tomato", "Tomato_healthy"),
    "Tomato leaf bacterial spot":
        ("Tomato", "Tomato_Bacterial_spot"),
    "Tomato leaf late blight":
        ("Tomato", "Tomato_Late_blight"),
    "Tomato leaf mosaic virus":
        ("Tomato", "Tomato__Tomato_mosaic_virus"),
    "Tomato leaf yellow virus":
        ("Tomato", None),
    "Tomato mold leaf":
        ("Tomato", "Tomato_Leaf_Mold"),
    "Tomato Septoria leaf spot":
        ("Tomato", "Tomato_Septoria_leaf_spot"),
    "Tomato two spotted spider mites leaf":
        (
            "Tomato",
            "Tomato_Spider_mites_Two_spotted_spider_mite"
        ),
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def crop_of(label):
    if label.startswith("Pepper__bell"):
        return "Pepper__bell"

    if label.startswith("Tomato"):
        return "Tomato"

    return label.split("___")[0]


def load_checkpoint(path, device):
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        ckpt = torch.load(
            path, map_location="cpu", weights_only=True
        )
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    names = ckpt.get("class_names")
    weights = ckpt.get("model_state_dict")

    if names is None or weights is None:
        raise ValueError(
            f"Checkpoint thieu class_names/model_state_dict: {path}"
        )

    names = list(names)

    model = get_mobilenet_v2_model(
        num_classes=len(names),
        pretrained=False
    )

    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()

    print(f"Da tai {len(names)} lop: {path.name}")
    return model, names


def read_images():
    records = []
    skipped = 0

    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        group = path.parent.name

        if group not in LABEL_MAP:
            print("Bo qua nhom chua anh xa:", group)
            skipped += 1
            continue

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as error:
            print("Anh loi:", path, error)
            skipped += 1
            continue

        crop, label = LABEL_MAP[group]

        records.append({
            "path": path,
            "group": group,
            "crop": crop,
            "label": label
        })

    print("Anh hop le:", len(records))
    print("Anh bo qua:", skipped)

    return records


def best_in_crop(logits, names, crop):
    indices = [
        i for i, name in enumerate(names)
        if crop_of(name) == crop
    ]

    if not indices:
        return ""

    best_idx = max(
        indices, key=lambda i: float(logits[i])
    )

    return names[best_idx]


def save_csv(path, rows, fieldnames):
    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_images", type=int, default=0
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )

    args = parser.parse_args()

    if not DATA_DIR.is_dir():
        raise FileNotFoundError(DATA_DIR)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Thiet bi:", device)

    model18, names18 = load_checkpoint(
        MODEL_18, device
    )
    model97, names97 = load_checkpoint(
        MODEL_97, device
    )

    assert len(names18) == 18
    assert len(names97) == 97

    records = read_images()

    if args.max_images > 0:
        records = records[:args.max_images]

    if not records:
        raise ValueError("Khong co anh de danh gia.")

    known_labels = set(names97)

    for record in records:
        label = record["label"]

        if label is not None and label not in known_labels:
            raise ValueError(
                f"Nhãn không tồn tại trong checkpoint: {label}"
            )

    results = []

    with torch.inference_mode():
        for start in range(
            0, len(records), args.batch_size
        ):
            batch = records[
                start:start + args.batch_size
            ]

            images = []

            for record in batch:
                with Image.open(record["path"]) as img:
                    img = ImageOps.exif_transpose(img)
                    images.append(
                        TRANSFORM(img.convert("RGB"))
                    )

            x = torch.stack(images).to(device)

            logits18 = model18(x)
            logits97 = model97(x)

            probs18 = logits18.softmax(dim=1)

            for j, record in enumerate(batch):
                true_crop = record["crop"]
                true_label = record["label"]

                top18 = torch.topk(
                    probs18[j], k=min(3, len(names18))
                )

                crop_pred = names18[
                    int(top18.indices[0].item())
                ]

                crop_pred = crop_of(crop_pred)

                top3 = " | ".join(
                    f"{names18[int(idx)]}: {float(score):.1%}"
                    for idx, score in zip(
                        top18.indices.tolist(),
                        top18.values.tolist()
                    )
                )

                raw97 = names97[
                    int(logits97[j].argmax().item())
                ]

                # Bệnh dự đoán khi biết đúng cây
                disease_oracle = best_in_crop(
                    logits97[j], names97, true_crop
                )

                # Bệnh dự đoán theo cây của mô hình 18
                disease_pipeline = best_in_crop(
                    logits97[j], names97, crop_pred
                )

                crop_ok = crop_pred == true_crop

                disease_ok = (
                    disease_oracle == true_label
                    if true_label is not None else None
                )

                pipeline_ok = (
                    crop_ok and disease_pipeline == true_label
                    if true_label is not None else None
                )

                results.append({
                    "path": str(record["path"]),
                    "nhom_plantdoc": record["group"],
                    "cay_that": true_crop,
                    "benh_theo_nhan_nguon": (
                        true_label or "CHUA_XAC_MINH"
                    ),
                    "cay_ai_18": crop_pred,
                    "dung_cay_18": int(crop_ok),
                    "top3_cay_18": top3,
                    "du_doan_97_tu_do": raw97,
                    "benh_97_khi_biet_dung_cay": disease_oracle,
                    "dung_benh_khi_biet_cay": (
                        "" if disease_ok is None
                        else int(disease_ok)
                    ),
                    "benh_du_doan_ket_hop": disease_pipeline,
                    "dung_ca_cay_va_benh": (
                        "" if pipeline_ok is None
                        else int(pipeline_ok)
                    )
                })

            print(
                f"Da xu ly: "
                f"{min(start + len(batch), len(records))}"
                f"/{len(records)}",
                flush=True
            )

    file_detail = OUTPUT / "DU_DOAN_TUNG_ANH.csv"

    save_csv(
        file_detail,
        results,
        list(results[0].keys())
    )

    groups = defaultdict(list)

    for row in results:
        groups[row["nhom_plantdoc"]].append(row)

    summary = []

    for group, rows in sorted(groups.items()):
        labeled = [
            r for r in rows
            if r["dung_ca_cay_va_benh"] != ""
        ]

        crop_correct = sum(
            int(r["dung_cay_18"]) for r in rows
        )

        disease_correct = sum(
            int(r["dung_benh_khi_biet_cay"])
            for r in labeled
        )

        pipeline_correct = sum(
            int(r["dung_ca_cay_va_benh"])
            for r in labeled
        )

        summary.append({
            "nhom": group,
            "so_anh": len(rows),
            "dung_cay": crop_correct,
            "ty_le_dung_cay": round(
                100 * crop_correct / len(rows), 2
            ),
            "so_anh_co_nhan_benh": len(labeled),
            "dung_benh_khi_biet_cay": disease_correct,
            "dung_ca_cay_va_benh": pipeline_correct
        })

    save_csv(
        OUTPUT / "BAO_CAO_THEO_NHOM.csv",
        summary,
        list(summary[0].keys())
    )

    labeled = [
        r for r in results
        if r["dung_ca_cay_va_benh"] != ""
    ]

    n = len(results)
    correct_crop = sum(
        int(r["dung_cay_18"]) for r in results
    )

    correct_disease = sum(
        int(r["dung_benh_khi_biet_cay"])
        for r in labeled
    )

    correct_pipeline = sum(
        int(r["dung_ca_cay_va_benh"])
        for r in labeled
    )

    print("\n========== KET QUA ==========")
    print("Tong anh:", n)
    print(
        f"Dung loai cay: {correct_crop}/{n} "
        f"({100 * correct_crop / n:.2f}%)"
    )
    print("Anh co nhan benh de cham:", len(labeled))

    if labeled:
        print(
            "Dung benh khi biet dung cay:",
            f"{correct_disease}/{len(labeled)}",
            f"({100 * correct_disease / len(labeled):.2f}%)"
        )

        print(
            "Dung ca cay va benh:",
            f"{correct_pipeline}/{len(labeled)}",
            f"({100 * correct_pipeline / len(labeled):.2f}%)"
        )

    print("\nDa xuat bao cao tai:", OUTPUT)


if __name__ == "__main__":
    main()
