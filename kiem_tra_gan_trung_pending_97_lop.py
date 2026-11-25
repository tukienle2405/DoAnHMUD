"""Kiem tra gan trung cac anh pending trong manifest 97 lop.

Chay: python -u kiem_tra_gan_trung_pending_97_lop.py
Phu thuoc: python -m pip install pillow numpy opencv-python

CHU Y: KHONG xoa/doi ten/di chuyen anh; KHONG chia train/val/test.
Ket qua la nhom NGHI GAN TRUNG, khong phai bang chung chung minh
cac anh cung mot la goc.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "dataset_review" / "dataset_97_lop"
INPUT = BASE / "du_lieu_sau_lam_sach" / "dataset_manifest_SAU_SHA.csv"
OUTPUT = BASE / "kiem_tra_gan_trung"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Nguong de SANG LOC; khong duoc coi la chung minh anh trung.
PHASH_MAX = 3
DHASH_MAX = 5
MAX_CANDIDATES_PER_IMAGE = 2500
MAX_PAIR_ROWS = 100000


def write_csv(filename, columns, data):
    with (OUTPUT / filename).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(data)


def hashes(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("L")
        im.load()
        small = np.asarray(im.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        dh = np.frombuffer(im.resize((9, 8), Image.Resampling.LANCZOS).tobytes(), dtype=np.uint8).reshape(8, 9)

    dct = cv2.dct(small)[:8, :8].reshape(-1)
    middle = float(np.median(dct[1:]))
    p = 0
    for v in dct:
        p = (p << 1) | int(v > middle)
    d = 0
    for a, b in zip(dh[:, :-1].reshape(-1), dh[:, 1:].reshape(-1)):
        d = (d << 1) | int(a > b)
    return p, d


def crop_of(label):
    if label.startswith("Tomato"):
        return "Tomato"
    if label.startswith("Pepper__bell"):
        return "Pepper"
    return label.split("___", 1)[0]


class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        while x != self.parent[x]:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(f"Khong tim thay file: {INPUT}")

    with INPUT.open(encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.DictReader(f))

    necessary = {"path", "class", "split", "status", "source", "group_id"}
    if not all_rows or not necessary.issubset(all_rows[0]):
        raise ValueError(f"Manifest thieu cot: {necessary - set(all_rows[0] if all_rows else {})}")

    pending = [(i, r) for i, r in enumerate(all_rows) if r["split"] == "pending"]
    print(f"Manifest: {len(all_rows):,} anh | Pending: {len(pending):,} anh", flush=True)

    valid = []
    errors = []
    for j, (original_index, row) in enumerate(pending, 1):
        try:
            p, d = hashes(Path(row["path"]))
            valid.append({"origin": original_index, "path": row["path"],
                          "label": row["class"], "crop": crop_of(row["class"]),
                          "phash": p, "dhash": d})
        except Exception as exc:
            errors.append([row["path"], row["class"], str(exc)[:300]])
        if j % 3000 == 0:
            print(f"Da xu ly {j:,}/{len(pending):,} anh", flush=True)

    print(f"Hash thanh cong: {len(valid):,}; Anh loi: {len(errors):,}", flush=True)

    # 4 dai 16-bit tren pHash. Neu khoang cach pHash <= 3,
    # theo nguyen ly pigeonhole se co it nhat mot dai khop hoan toan.
    # Khi bucket qua dong, chung ta ghi canh bao, KHONG tu ket luan du lieu an toan.
    dsu = DSU(len(valid))
    index = defaultdict(list)
    near_pairs = []
    total_matched_pairs = 0
    overflow = []
    for i, row in enumerate(valid):
        p, d, label = row["phash"], row["dhash"], row["label"]
        matches = set()
        overloaded = False
        for band in range(4):
            key = (label, band, (p >> (16 * band)) & 0xFFFF)
            bucket = index[key]
            if len(bucket) > MAX_CANDIDATES_PER_IMAGE:
                overloaded = True
            else:
                matches.update(bucket)
        if overloaded:
            overflow.append([row["path"], label,
                             "Bucket pHash qua dong; ket qua nhom co the thieu"])
        for j in matches:
            other = valid[j]
            pdist = (p ^ other["phash"]).bit_count()
            if pdist > PHASH_MAX:
                continue
            ddist = (d ^ other["dhash"]).bit_count()
            if ddist > DHASH_MAX:
                continue
            dsu.union(i, j)
            total_matched_pairs += 1
            if len(near_pairs) < MAX_PAIR_ROWS:
                near_pairs.append([label, pdist, ddist, other["path"], row["path"]])
        for band in range(4):
            index[(label, band, (p >> (16 * band)) & 0xFFFF)].append(i)

    groups = defaultdict(list)
    for i, row in enumerate(valid):
        groups[(row["label"], dsu.find(i))].append(i)

    # Neu pHash VA dHash deu TRUNG CHINH XAC tren hai nhan khac nhau,
    # ghi nhan nghi van, TUYET DOI KHONG tu xoa hay gan lai nhan.
    by_both = defaultdict(list)
    for i, r in enumerate(valid):
        by_both[(r["crop"], r["phash"], r["dhash"])].append(i)
    cross = []
    for (_, p, d), indexes in by_both.items():
        if len({valid[i]["label"] for i in indexes}) < 2:
            continue
        for i in indexes:
            r = valid[i]
            cross.append([f"{p:016x}", f"{d:016x}", r["label"], r["path"]])

    # Gan group_id moi cho pending; khong thay doi split, status hay group_id cua Tea/Durian.
    near_id = {}
    summaries = []
    large_groups = []
    for (label, root), items in groups.items():
        name = f"N{min(items):07d}"
        for i in items:
            near_id[valid[i]["origin"]] = name
        summaries.append([label, name, len(items), valid[items[0]]["path"]])
        if len(items) >= 20:
            large_groups.append([label, name, len(items), valid[items[0]]["path"]])

    outfile = OUTPUT / "dataset_manifest_NHOM_SO_BO.csv"
    fields = list(all_rows[0]) + ["near_group_id", "near_review"]
    with outfile.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        error_paths = {p for p, _, _ in errors}
        cross_paths = {p for _, _, _, p in cross}
        overflow_paths = {p for p, _, _ in overflow}
        for i, record in enumerate(all_rows):
            row = dict(record)
            row["near_group_id"] = near_id.get(i, "")
            if record["split"] != "pending":
                row["near_review"] = "DA_CO_SPLIT_GIU_NGUYEN"
            elif record["path"] in error_paths:
                row["near_review"] = "LOI_ANH"
            elif record["path"] in overflow_paths:
                row["near_review"] = "CAN_XEM_BUCKET_DONG"
            elif record["path"] in cross_paths:
                row["near_review"] = "CAN_XEM_KHAC_NHAN"
            else:
                row["near_review"] = "NHOM_UNG_VIEN"
            writer.writerow(row)

    write_csv("nhom_gan_trung_cung_nhan.csv",
              ["class", "near_group_id", "image_count", "example_path"],
              sorted(summaries))
    write_csv("cap_gan_trung_cung_nhan.csv",
              ["class", "phash_distance", "dhash_distance", "path1", "path2"],
              near_pairs)
    write_csv("nghi_gan_trung_khac_nhan.csv",
              ["phash", "dhash", "class", "path"], cross)
    write_csv("nhom_qua_lon_can_xem.csv",
              ["class", "near_group_id", "image_count", "example_path"],
              sorted(large_groups, key=lambda x: -x[2]))
    write_csv("loi_anh_pending.csv", ["path", "class", "error"], errors)
    write_csv("bucket_qua_dong.csv", ["path", "class", "reason"], overflow)

    counts = Counter(len(v) for v in groups.values())
    report = [
        "BAO CAO KIEM TRA ANH GAN TRUNG - PENDING 97 LOP",
        "=" * 58,
        f"So anh trong manifest: {len(all_rows):,}",
        f"Pending: {len(pending):,}",
        f"Anh pending tinh hash thanh cong: {len(valid):,}",
        f"Anh loi: {len(errors):,}",
        f"So nhom ung vien cung nhan: {len(groups):,}",
        f"Nhom co >=2 anh: {sum(n for sz,n in counts.items() if sz>=2):,}",
        f"Anh nam trong nhom >=2: {sum(sz*n for sz,n in counts.items() if sz>=2):,}",
        f"Cap gan trung cung nhan: {total_matched_pairs:,}",
        f"So dong cap da ghi CSV: {len(near_pairs):,}",
        f"Nhom lon >=20 anh: {len(large_groups):,}",
        f"File gap bucket qua dong: {len(overflow):,}",
        f"Anh nghi gan trung KHAC NHAN (pHash+dHash deu bang nhau): {len(cross):,}",
        "",
        "LUU Y QUAN TRONG:",
        "1. CHUA chia train/val/test cho pending.",
        "2. Nhung nhom nay CHI la goi y dua tren hash, KHONG chac chan cung la goc.",
        "3. Hash khong phat hien day du anh xoay/lat va cac kieu augmentation khac.",
        "4. Neu co bucket qua dong, viec phat hien co the bi bo sot.",
        "5. Chua xac minh nhan Papaya, Guava Dot, Sugarcane Sett_rot.",
        "6. Khong sua/xoa/di chuyen anh goc; split Tea va Durian giu nguyen.",
    ]
    report_text = "\n".join(report)
    (OUTPUT / "bao_cao_gan_trung_pending.txt").write_text(report_text, encoding="utf-8")
    print("\n" + report_text)
    print("\nKet qua:", OUTPUT)


if __name__ == "__main__":
    main()
