# -*- coding: utf-8 -*-
"""
KIEM TRA TOP 3 CAY + MO HINH 97 LOP - KHONG HUAN LUYEN LAI.

Dat file nay tai D:\\DoanHMvUD\\kiem_tra_top3_18_97.py

Phan tich CSV (khong can model / GPU):
    py -3 kiem_tra_top3_18_97.py --mode audit

Chay 2 model tren cung 238 anh Test da khoa:
    py -3 -u kiem_tra_top3_18_97.py --batch_size 8

Co the chi ro duong dan khi can:
    py -3 -u kiem_tra_top3_18_97.py --csv "duong_dan\\DU_DOAN_TUNG_ANH.csv" \
        --model18 "duong_dan\\best_field_candidate.pth" \
        --model97 "duong_dan\\best_plant_model.pth"

Chu y:
- Khong ghi de checkpoint va khong huan luyen.
- Dung CHINH XAC nhung anh da co trong DU_DOAN_TUNG_ANH.csv.
- CHUA_XAC_MINH duoc giu ngoai mau so tinh do chinh xac benh.
- Phuong an Top3 chinh: P18(cay) * P97(benh | cay, anh).
- Diem ket hop chi de xep hang, KHONG phai xac suat da hieu chuan.
- Phuong an top3_global la doi chung; khong tu dong chon theo ket qua Test.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UNVERIFIED = "CHUA_XAC_MINH"
EXPECTED_CROPS = set([
    "Banana", "Cashew", "Chili", "Coffee", "Corn", "Durian",
    "Guava", "Jackfruit", "Lemon", "Mango", "Papaya", "Pepper__bell",
    "Potato", "Rice", "Rubber", "Sugarcane", "Tea", "Tomato"
])


def crop_of(class_name):
    """Quy uoc ten nhan 97 lop cua DU AN hien tai."""
    if class_name.startswith("Pepper__bell"):
        return "Pepper__bell"
    if class_name.startswith("Tomato"):
        return "Tomato"
    return class_name.split("___", 1)[0]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def find_input_csv(explicit=None):
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Khong thay CSV: {path}")
        return path
    candidates = [
        ROOT / "external_field_pool_v2" / "field_adaptation_v1" / "KET_QUA_KET_HOP_18_97" / "DU_DOAN_TUNG_ANH.csv",
        ROOT / "DU_DOAN_TUNG_ANH.csv",
        ROOT / "DU_DOAN_TUNG_ANH(1).csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Khong thay DU_DOAN_TUNG_ANH.csv. Dat CSV tai thu muc D:\\DoanHMvUD "
        "hoac them --csv duong_dan_file.csv"
    )


def parse_top3(value):
    """Dinh dang: 'Tomato: 78.5% | Potato: 15.2% | Corn: 1.1%'"""
    result = []
    for part in value.split(" | "):
        if ": " not in part:
            raise ValueError(f"Top3 sai dinh dang: {value}")
        crop, prob = part.rsplit(": ", 1)
        result.append((crop, float(prob.rstrip("%")) / 100.0))
    if len(result) != 3 or len({x[0] for x in result}) != 3:
        raise ValueError(f"Top3 phai co dung 3 cay khac nhau: {value}")
    return result


def is_verified(row):
    return bool(row["benh_theo_nhan_nguon"]) and row["benh_theo_nhan_nguon"] != UNVERIFIED


def pct(n, d):
    return round(100.0 * n / d, 2) if d else None


def check_csv(rows):
    required = [
        "path", "nhom_plantdoc", "cay_that", "benh_theo_nhan_nguon",
        "cay_ai_18", "dung_cay_18", "top3_cay_18",
        "benh_97_khi_biet_dung_cay", "benh_du_doan_ket_hop",
        "dung_ca_cay_va_benh"
    ]
    if not rows:
        raise ValueError("CSV khong co anh nao")
    missing = [x for x in required if x not in rows[0]]
    if missing:
        raise ValueError(f"Thieu cot: {missing}")
    paths = [x["path"].casefold() for x in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("CSV co path bi trung!")
    for row in rows:
        parse_top3(row["top3_cay_18"])
        if row["cay_that"] not in EXPECTED_CROPS:
            raise ValueError(f"Cay ngoai danh sach 18 cay: {row['cay_that']}")


def check_locked_test(rows, explicit=None):
    path = Path(explicit) if explicit else ROOT / "external_field_pool_v2" / "EXTERNAL_SPLIT_LOCKED.csv"
    if not path.is_file():
        print("[CANH BAO] Khong thay EXTERNAL_SPLIT_LOCKED.csv de doi chieu.")
        print("           Van chi test tren anh trong CSV ket qua cu.")
        return False
    locked = read_csv(path)
    allowed = {r["path"].casefold() for r in locked
               if r.get("split") == "test" and r.get("reason") == "OK"}
    submitted = {r["path"].casefold() for r in rows}
    if allowed != submitted:
        raise ValueError(
            f"CSV khong trung Test da khoa! Locked={len(allowed)}, "
            f"CSV={len(submitted)}, khac={len(allowed.symmetric_difference(submitted))}"
        )
    print(f"[OK] Trung khop 100% danh sach Test da khoa: {len(rows)} anh")
    return True


def analyze_existing(rows, outdir):
    """Phan tich tu CSV san co; KHONG gia lap ket qua Top3."""
    check_csv(rows)
    n = len(rows)
    labeled = [r for r in rows if is_verified(r)]
    unverified = [r for r in rows if not is_verified(r)]
    top3 = {r["path"]: {c for c, _ in parse_top3(r["top3_cay_18"])} for r in rows}
    in_top3 = lambda r: r["cay_that"] in top3[r["path"]]

    correct_crop = sum(int(r["dung_cay_18"]) for r in rows)
    correct_old = sum(int(r["dung_ca_cay_va_benh"]) for r in labeled)
    correct_oracle = sum(int(r["dung_benh_khi_biet_cay"]) for r in labeled)
    crop_top3 = sum(in_top3(r) for r in rows)
    label_top3 = sum(in_top3(r) for r in labeled)
    oracle_top3 = sum(in_top3(r) and int(r["dung_benh_khi_biet_cay"]) for r in labeled)

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["nhom_plantdoc"]].append(r)
    by_group = []
    for name, group in sorted(grouped.items()):
        known = [r for r in group if is_verified(r)]
        by_group.append({
            "nhom": name, "tong_anh": len(group),
            "dung_cay_top1": sum(int(r["dung_cay_18"]) for r in group),
            "cay_that_trong_top3": sum(in_top3(r) for r in group),
            "co_nhan_benh": len(known),
            "dung_ca_cay_va_benh_cu": sum(int(r["dung_ca_cay_va_benh"]) for r in known),
            "dung_benh_khi_biet_cay": sum(int(r["dung_benh_khi_biet_cay"]) for r in known),
            "gioi_han_top3_voi_benh_hien_tai": sum(
                in_top3(r) and int(r["dung_benh_khi_biet_cay"]) for r in known
            ),
        })

    confusions = Counter((r["benh_theo_nhan_nguon"],
                          r["benh_97_khi_biet_dung_cay"])
                         for r in labeled if r["cay_that"] == "Tomato")
    rows_conf = [{"nhan_that": a, "du_doan_khi_biet_dung_cay": b, "so_anh": v}
                 for (a, b), v in sorted(confusions.items())]
    tomato = [r for r in labeled if r["cay_that"] == "Tomato"]
    by_tomato_disease = []
    for disease in sorted({r["benh_theo_nhan_nguon"] for r in tomato}):
        group = [r for r in tomato if r["benh_theo_nhan_nguon"] == disease]
        by_tomato_disease.append({
            "benh": disease, "so_anh": len(group),
            "dung_cay_top1": sum(int(r["dung_cay_18"]) for r in group),
            "dung_benh_biet_cay": sum(int(r["dung_benh_khi_biet_cay"]) for r in group),
            "dung_ca_cay_benh": sum(int(r["dung_ca_cay_va_benh"]) for r in group),
        })

    summary = {
        "nguon": "DU_DOAN_TUNG_ANH.csv, du lieu co san, khong suy dien ket qua model moi",
        "tong_anh": n, "anh_co_nhan_benh": len(labeled),
        "anh_chua_xac_minh": len(unverified),
        "crop_top1_correct": correct_crop, "crop_top1_accuracy_pct": pct(correct_crop, n),
        "crop_top3_recall": crop_top3, "crop_top3_recall_pct": pct(crop_top3, n),
        "crop_top3_recall_labeled": label_top3,
        "crop_top3_recall_labeled_pct": pct(label_top3, len(labeled)),
        "joint_old_correct": correct_old,
        "joint_old_accuracy_pct": pct(correct_old, len(labeled)),
        "disease_true_crop_correct": correct_oracle,
        "disease_true_crop_accuracy_pct": pct(correct_oracle, len(labeled)),
        "upper_bound_top3_with_fixed_disease_model": oracle_top3,
        "upper_bound_top3_pct": pct(oracle_top3, len(labeled)),
        "maximum_extra_correct_if_choosing_true_top3_crop": oracle_top3 - correct_old,
        "NOTE": "Gioi han Top3 dung nhan that (oracle), KHONG PHAI ket qua cua thuat toan Top3 moi.",
    }
    save_json(outdir / "PHAN_TICH_CSV_HIEN_TAI.json", summary)
    write_csv(outdir / "NHOM_HIEN_TAI.csv", by_group)
    write_csv(outdir / "NHAM_LAN_BENH_CA_CHUA.csv", rows_conf,
              ["nhan_that", "du_doan_khi_biet_dung_cay", "so_anh"])
    write_csv(outdir / "BENH_CA_CHUA_THEO_LOP.csv", by_tomato_disease)
    write_csv(outdir / "ANH_CHUA_XAC_MINH.csv", unverified, list(rows[0]))
    write_csv(outdir / "ANH_CO_THE_CUU_BANG_TOP3.csv", [
        {**r, "cay_that_co_trong_top3": int(in_top3(r))}
        for r in labeled
        if int(r["dung_cay_18"]) == 0 and in_top3(r) and int(r["dung_benh_khi_biet_cay"]) == 1
    ])
    print("\n===== KIEM TRA TU CSV (CHUA CHAY TOP3) =====")
    for k, v in summary.items():
        if k not in ("nguon", "NOTE"):
            print(f"{k}: {v}")
    print("Thu muc phan tich:", outdir)
    return summary


def newest(paths):
    paths = list(paths)
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def resolve_model18(explicit):
    if explicit:
        p = Path(explicit)
    else:
        p = newest((ROOT / "training_runs").glob(
            "crop18_field_pilot_*/best_field_candidate.pth"
        ))
    if p is None or not p.is_file():
        raise FileNotFoundError(
            "Khong tim thay checkpoint 18 cay da fine-tune field "
            "(crop18_field_pilot_*/best_field_candidate.pth). "
            "Hay dung --model18 duong_dan_cu_the; KHONG tu dong dung model cu."
        )
    return p


def resolve_model97(explicit):
    if explicit:
        p = Path(explicit)
    else:
        paths = list((ROOT / "training_runs").glob(
            "plant97_full_*/best_plant_model.pth"
        ))
        paths += list((ROOT / "training_runs").glob(
            "plant97_*/best_plant_model.pth"
        ))
        p = newest(paths)
    if p is None or not p.is_file():
        raise FileNotFoundError("Khong tim thay checkpoint 97 lop; dung --model97")
    return p


def load_model(path, nclasses, device):
    # import tai day de --mode audit khong yeu cau PyTorch
    import torch
    from model_architecture import get_mobilenet_v2_model

    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or "model_state_dict" not in data or "class_names" not in data:
        raise ValueError(f"Checkpoint sai cau truc: {path}")
    classes = list(data["class_names"])
    if len(classes) != nclasses or len(set(classes)) != nclasses:
        raise ValueError(f"Checkpoint {path.name} khong co dung {nclasses} nhan duy nhat")
    if nclasses == 18 and set(classes) != EXPECTED_CROPS:
        raise ValueError("Checkpoint 18 cay co nhan khac voi 18 cay cua du an")
    if nclasses == 97:
        other = set(map(crop_of, classes)) - EXPECTED_CROPS
        if other:
            raise ValueError(f"97 lop co ten cay la: {sorted(other)}")
    model = get_mobilenet_v2_model(num_classes=nclasses, pretrained=False)
    model.load_state_dict(data["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, classes


def best_disease_for_crop(p97, indices):
    """Tra ve (index benh, p97 toan cuc, p97 dieu kien theo cay)."""
    # p97 la list float, khong can torch o buoc ket hop
    if not indices:
        raise ValueError("Mot cay khong co lop benh nao trong 97 lop")
    best = max(indices, key=lambda j: p97[j])
    total = sum(p97[j] for j in indices)
    cond = p97[best] / total if total > 0 else 0.0
    return best, p97[best], cond


def rank_candidates(p18, p97, crops18, classes97, top_k=3):
    """2 cach ket hop DANG CO DINH, khong tuning tren Test."""
    crop_to_indices = defaultdict(list)
    for i, name in enumerate(classes97):
        crop_to_indices[crop_of(name)].append(i)
    crop_index = sorted(range(len(crops18)), key=lambda j: p18[j], reverse=True)[:top_k]
    results = []
    for i in crop_index:
        crop = crops18[i]
        label_i, global_p, cond_p = best_disease_for_crop(p97, crop_to_indices[crop])
        results.append({
            "crop": crop, "crop_p": p18[i], "disease": classes97[label_i],
            "disease_p_global": global_p, "disease_p_given_crop": cond_p,
            "score_cond": p18[i] * cond_p,
            "score_global": p18[i] * global_p,
        })
    return sorted(results, key=lambda r: r["score_cond"], reverse=True), sorted(
        results, key=lambda r: r["score_global"], reverse=True
    )


def inference(rows, model18path, model97path, outdir, batch_size, device_text):
    import torch
    from PIL import Image, ImageOps
    from torchvision import transforms

    if batch_size < 1:
        raise ValueError("batch_size phai >= 1")
    # Tat ca anh phai ton tai, khong duoc ngam bo anh kho => sai mau so.
    missing = [r["path"] for r in rows if not Path(r["path"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Khong mo duoc {len(missing)}/{len(rows)} anh. "
            f"Anh dau tien: {missing[0]}"
        )
    if device_text == "auto":
        device_text = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_text)
    print("Thiet bi:", device)
    print("Model 18:", model18path)
    print("Model 97:", model97path)
    model18, names18 = load_model(model18path, 18, device)
    model97, names97 = load_model(model97path, 97, device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    results = []
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset:offset + batch_size]
        tensors = []
        for r in batch_rows:
            with Image.open(r["path"]) as im:
                img = ImageOps.exif_transpose(im).convert("RGB")
                tensors.append(transform(img))
        x = torch.stack(tensors).to(device)
        with torch.inference_mode():
            p18_batch = torch.softmax(model18(x), dim=1).cpu().tolist()
            p97_batch = torch.softmax(model97(x), dim=1).cpu().tolist()
        for original, p18, p97 in zip(batch_rows, p18_batch, p97_batch):
            ranked_cond, ranked_global = rank_candidates(p18, p97, names18, names97)
            top1_crop = names18[max(range(18), key=lambda j: p18[j])]
            top1_disease, _, _ = best_disease_for_crop(
                p97, [j for j, name in enumerate(names97) if crop_of(name) == top1_crop]
            )
            # Toi uu tinh toan: dung '97 lop boi cung 1 lan forward'.
            # Dau ra baseline tai tinh lai, KHONG phu thuoc du doan cu.
            pred_old = names97[top1_disease]
            gt = original["benh_theo_nhan_nguon"]
            verified = is_verified(original)
            correct_base = int(pred_old == gt) if verified else ""
            correct_cond = int(ranked_cond[0]["disease"] == gt) if verified else ""
            correct_global = int(ranked_global[0]["disease"] == gt) if verified else ""
            results.append({
                **original,
                "cay_18_chay_lai": top1_crop,
                "top3_cay_18_chay_lai": " | ".join(
                    f"{names18[j]}: {100*p18[j]:.2f}%"
                    for j in sorted(range(18), key=lambda j: p18[j], reverse=True)[:3]
                ),
                "benh_top1_chay_lai": pred_old,
                "dung_top1_chay_lai": correct_base,
                "cay_top3_chon": ranked_cond[0]["crop"],
                "benh_top3_chon": ranked_cond[0]["disease"],
                "diem_ket_hop_top3": round(ranked_cond[0]["score_cond"], 8),
                "dung_top3": correct_cond,
                "danh_sach_ung_vien_top3": " | ".join(
                    f"{a['crop']} => {a['disease']} (diem={a['score_cond']:.6f})"
                    for a in ranked_cond
                ),
                "cay_top3_global": ranked_global[0]["crop"],
                "benh_top3_global": ranked_global[0]["disease"],
                "dung_top3_global": correct_global,
            })
        print(f"Da du doan: {min(offset+batch_size,len(rows))}/{len(rows)} anh", flush=True)
    write_csv(outdir / "DU_DOAN_TOP3_TUNG_ANH.csv", results)

    labeled = [r for r in results if is_verified(r)]
    base = sum(int(r["dung_top1_chay_lai"]) for r in labeled)
    fusion = sum(int(r["dung_top3"]) for r in labeled)
    diagnostic = sum(int(r["dung_top3_global"]) for r in labeled)
    old = sum(int(r["dung_ca_cay_va_benh"]) for r in labeled)
    mismatch_crops = sum(r["cay_18_chay_lai"] != r["cay_ai_18"] for r in results)
    mismatch_disease = sum(r["benh_top1_chay_lai"] != r["benh_du_doan_ket_hop"]
                           for r in results)

    summary = {
        "so_anh": len(results), "co_nhan_benh": len(labeled),
        "chua_xac_minh": len(results) - len(labeled),
        "baseline_csv_cu": {"dung": old, "accuracy_pct": pct(old, len(labeled))},
        "baseline_top1_chay_lai": {"dung": base, "accuracy_pct": pct(base, len(labeled))},
        "top3_conditional_PRIMARY": {"dung": fusion, "accuracy_pct": pct(fusion, len(labeled))},
        "top3_global_DIAGNOSTIC": {"dung": diagnostic, "accuracy_pct": pct(diagnostic, len(labeled))},
        "tang_so_anh_dung_top3_so_voi_top1_chay_lai": fusion-base,
        "mismatch_crop_so_voi_CSV_cu": mismatch_crops,
        "mismatch_benh_top1_so_voi_CSV_cu": mismatch_disease,
        "NOTE": "Test chi de bao cao, khong tu dong lua chon/top tuning theo Test. "
                "Can validation doc lap truoc khi tich hop vao Streamlit.",
    }
    grouped = defaultdict(list)
    for r in results:
        grouped[r["nhom_plantdoc"]].append(r)
    report_groups = []
    for group, values in sorted(grouped.items()):
        known = [r for r in values if is_verified(r)]
        report_groups.append({
            "nhom": group, "so_anh": len(values), "so_anh_nhan_benh": len(known),
            "dung_top1_chay_lai": sum(int(r["dung_top1_chay_lai"]) for r in known),
            "dung_top3": sum(int(r["dung_top3"]) for r in known),
            "dung_top3_global_doi_chung": sum(int(r["dung_top3_global"]) for r in known),
        })
    write_csv(outdir / "BAO_CAO_TOP3_THEO_NHOM.csv", report_groups)
    write_csv(outdir / "CA_CHUA_TOP3_CON_SAI.csv", [
        r for r in labeled if r["cay_that"] == "Tomato" and int(r["dung_top3"]) == 0
    ], list(results[0]))
    save_json(outdir / "TONG_KET_TOP3.json", summary)
    print("\n===== KET QUA CHAY TOP 3 TREN TEST =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if mismatch_crops or mismatch_disease:
        print("[CANH BAO] Checkpoint/transform da khac lan chay CSV cu, "
              "khong duoc quy chenh lech so voi CSV cu cho Top3!")
    if fusion <= base:
        print("[KET LUAN] Top3 khong cai thien so voi Top1 chay lai. "
              "KHONG tich hop vao app.")
    else:
        print("[KET LUAN] Co cai thien so voi Top1 chay lai tren tap Test. "
              "Can kiem tra Validation doc lap truoc khi tich hop vao app.")
    print("Da luu bao cao:", outdir)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Kiem tra Top 3 cay + 97 benh")
    parser.add_argument("--mode", choices=["audit", "infer"], default="infer")
    parser.add_argument("--csv", default=None, help="DU_DOAN_TUNG_ANH.csv cu")
    parser.add_argument("--lock_csv", default=None, help="EXTERNAL_SPLIT_LOCKED.csv")
    parser.add_argument("--model18", default=None, help="best_field_candidate.pth")
    parser.add_argument("--model97", default=None, help="best_plant_model.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", default=None, help="Thu muc xuat CSV va JSON")
    args = parser.parse_args()
    in_csv = find_input_csv(args.csv)
    rows = read_csv(in_csv)
    check_csv(rows)
    outdir = Path(args.out) if args.out else (
        ROOT / "external_field_pool_v2" / "field_adaptation_v1" / "KET_QUA_TOP3_18_97"
    )
    print("CSV dau vao:", in_csv)
    print("Anh trong CSV:", len(rows))
    check_locked_test(rows, args.lock_csv)
    analyze_existing(rows, outdir)
    if args.mode == "audit":
        print("\n[OK] Da phan tich CSV; chua chay 2 model.")
        return
    model18 = resolve_model18(args.model18)
    model97 = resolve_model97(args.model97)
    inference(rows, model18, model97, outdir, args.batch_size, args.device)


if __name__ == "__main__":
    main()
