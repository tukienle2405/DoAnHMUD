
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import csv
import random
import sqlite3
import time

from PIL import Image, ImageOps
import imagehash


ROOT = Path(r"D:\DoanHMvUD")

MANIFEST = (
    ROOT / "dataset_review" / "dataset_97_lop"
    / "split_97_lop" / "dataset_manifest_SPLIT_DU_KIEN.csv"
)

EXTERNAL = (
    ROOT / "external_field_pool_v2"
    / "PlantDoc_khong_trung_SHA256"
)

OUTPUT = ROOT / "external_field_pool_v2" / "field_adaptation_v1"

LOCKED_CSV = OUTPUT / "EXTERNAL_SPLIT_LOCKED.csv"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SEED = 20261125
PHASH_THRESHOLD = 6

GROUP_TO_CROP = {
    "Bell_pepper leaf": "Pepper__bell",
    "Bell_pepper leaf spot": "Pepper__bell",

    "Corn Gray leaf spot": "Corn",
    "Corn leaf blight": "Corn",
    "Corn rust leaf": "Corn",

    "Potato leaf early blight": "Potato",
    "Potato leaf late blight": "Potato",

    "Tomato Early blight leaf": "Tomato",
    "Tomato leaf": "Tomato",
    "Tomato leaf bacterial spot": "Tomato",
    "Tomato leaf late blight": "Tomato",
    "Tomato leaf mosaic virus": "Tomato",
    "Tomato leaf yellow virus": "Tomato",
    "Tomato mold leaf": "Tomato",
    "Tomato Septoria leaf spot": "Tomato",
    "Tomato two spotted spider mites leaf": "Tomato",
}


def fmt(seconds):
    seconds = max(0, int(seconds))
    h, remain = divmod(seconds, 3600)
    m, s = divmod(remain, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def progress(title, completed, total, start):
    elapsed = time.perf_counter() - start
    speed = completed / max(elapsed, 0.001)
    eta = (total - completed) / max(speed, 0.001)

    finish = datetime.now() + timedelta(seconds=eta)

    print(
        f"{title}: {completed}/{total} "
        f"({completed / max(total, 1):.1%}) | "
        f"Da chay {fmt(elapsed)} | "
        f"Con lai ~{fmt(eta)} | "
        f"Xong luc {finish:%H:%M:%S}",
        flush=True
    )


class HashCache:
    def __init__(self, path):
        self.db = sqlite3.connect(path)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS hashes (
                path TEXT PRIMARY KEY,
                size INTEGER,
                mtime INTEGER,
                hash TEXT
            )
        """)

        self.saved = {
            row[0]: (row[1], row[2], int(row[3], 16))
            for row in self.db.execute(
                "SELECT path, size, mtime, hash FROM hashes"
            )
        }

        self.pending = 0

    def get(self, path):
        stat = path.stat()
        key = str(path)

        old = self.saved.get(key)

        if old is not None:
            if old[0] == stat.st_size and old[1] == stat.st_mtime_ns:
                return old[2]

        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            result = int(str(imagehash.phash(img)), 16)

        self.db.execute(
            "INSERT OR REPLACE INTO hashes VALUES (?, ?, ?, ?)",
            (
                key,
                stat.st_size,
                stat.st_mtime_ns,
                f"{result:016x}"
            )
        )

        self.saved[key] = (
            stat.st_size,
            stat.st_mtime_ns,
            result
        )

        self.pending += 1

        if self.pending >= 500:
            self.db.commit()
            self.pending = 0

        return result

    def close(self):
        self.db.commit()
        self.db.close()


class NearIndex:
    """
    Tim anh co pHash cach nhau <= 6 bit.
    Chi so 8 bang giup tranh so tung cap anh.
    """

    def __init__(self):
        self.bands = [
            defaultdict(list) for _ in range(8)
        ]

        self.hashes = {}

    def add(self, index, value):
        self.hashes[index] = value

        for band in range(8):
            part = (value >> (band * 8)) & 255
            self.bands[band][part].append(index)

    def search(self, value, threshold=6):
        candidates = set()

        for band in range(8):
            part = (value >> (band * 8)) & 255
            candidates.update(self.bands[band].get(part, []))

        matches = []

        for index in candidates:
            distance = (
                self.hashes[index] ^ value
            ).bit_count()

            if distance <= threshold:
                matches.append((index, distance))

        return sorted(matches, key=lambda x: x[1])


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a != b:
            self.parent[b] = a


def main():
    if LOCKED_CSV.exists():
        print("DA CO TAP KIEM THU DUOC KHOA:")
        print(LOCKED_CSV)
        print("Khong ghi de de tranh ro ri du lieu.")
        return

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)

    if not EXTERNAL.is_dir():
        raise FileNotFoundError(EXTERNAL)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    cache = HashCache(OUTPUT / "phash_cache.sqlite")

    with MANIFEST.open(
        encoding="utf-8-sig", newline=""
    ) as f:
        reference_rows = [
            row for row in csv.DictReader(f)
            if row["split"] in {"train", "val", "test"}
        ]

    print("Anh goc trong manifest:", len(reference_rows))

    reference_index = NearIndex()
    reference_paths = {}
    seen_hashes = set()

    started = time.perf_counter()

    # Quet cac anh ma model co the da tiep xuc
    for number, row in enumerate(reference_rows, 1):
        path = Path(row["path"])

        if not path.is_file():
            raise FileNotFoundError(
                f"Anh trong manifest bi thieu: {path}"
            )

        value = cache.get(path)

        # Mot gia tri pHash chi can luu mot dai dien
        if value not in seen_hashes:
            seen_hashes.add(value)
            key = len(reference_paths)
            reference_paths[key] = str(path)
            reference_index.add(key, value)

        if number % 10000 == 0:
            progress(
                "Quet du lieu goc",
                number,
                len(reference_rows),
                started
            )

    print("DA QUET XONG DU LIEU GOC.")

    paths = sorted(
        p for p in EXTERNAL.rglob("*")
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
    )

    print("Anh PlantDoc ban dau:", len(paths))

    records = []
    started_external = time.perf_counter()

    for number, path in enumerate(paths, 1):
        group = path.parent.name

        if group not in GROUP_TO_CROP:
            raise ValueError(
                f"Nhom PlantDoc chua duoc khai bao: {group}"
            )

        value = cache.get(path)

        near_old = reference_index.search(
            value, PHASH_THRESHOLD
        )

        nearest_path = ""
        nearest_distance = ""

        if near_old:
            match_id, distance = near_old[0]

            nearest_path = reference_paths[match_id]
            nearest_distance = distance

        records.append({
            "path": str(path),
            "group": group,
            "crop": GROUP_TO_CROP[group],
            "phash": f"{value:016x}",
            "hash_int": value,
            "near_old_path": nearest_path,
            "near_old_distance": nearest_distance,
            "split": "exclude",
            "reason": "",
            "cluster": ""
        })

        if number % 100 == 0 or number == len(paths):
            progress(
                "Kiem tra anh thuc dia",
                number,
                len(paths),
                started_external
            )

    # Gom cac anh PlantDoc gan giong nhau thanh nhom
    finder = UnionFind(len(records))
    external_index = NearIndex()

    for i, record in enumerate(records):
        matches = external_index.search(
            record["hash_int"],
            PHASH_THRESHOLD
        )

        for j, _ in matches:
            finder.union(i, j)

        external_index.add(i, record["hash_int"])

    clusters = defaultdict(list)

    for i in range(len(records)):
        root = finder.find(i)
        clusters[root].append(i)
        records[i]["cluster"] = f"P{root:05d}"

    eligible = defaultdict(list)

    for members in clusters.values():
        crops = {
            records[i]["crop"] for i in members
        }

        has_old_match = any(
            records[i]["near_old_path"]
            for i in members
        )

        if len(crops) > 1:
            for i in members:
                records[i]["reason"] = "NEAR_CROSS_CROP"
            continue

        if has_old_match:
            for i in members:
                records[i]["reason"] = "NEAR_OLD_DATA"
            continue

        # Mot cluster phai nam trong cung mot split
        main_group = sorted(
            {records[i]["group"] for i in members}
        )[0]

        eligible[main_group].append(members)

    # Chia theo nhom anh, khong chia tung anh
    rng = random.Random(SEED)

    for group, cluster_list in sorted(eligible.items()):
        rng.shuffle(cluster_list)

        n = len(cluster_list)

        n_test = max(1, round(n * 0.20))
        n_val = max(1, round(n * 0.20)) if n >= 2 else 0

        if n_test + n_val > n:
            n_val = n - n_test

        for j, members in enumerate(cluster_list):
            if j < n_test:
                split = "test"
            elif j < n_test + n_val:
                split = "val"
            else:
                split = "train"

            for i in members:
                records[i]["split"] = split
                records[i]["reason"] = "OK"

    fieldnames = [
        "path",
        "group",
        "crop",
        "split",
        "reason",
        "phash",
        "near_old_path",
        "near_old_distance",
        "cluster"
    ]

    # Ghi tap du lieu khoa, khong xoa anh
    with LOCKED_CSV.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:
            writer.writerow({
                k: record[k] for k in fieldnames
            })

    cache.close()

    counts = Counter(
        (r["crop"], r["split"]) for r in records
    )

    print("\n========== KET QUA ==========")

    for crop in sorted(set(GROUP_TO_CROP.values())):
        values = [
            counts[(crop, split)]
            for split in ("train", "val", "test", "exclude")
        ]

        print(
            f"{crop:15s} | "
            f"Train {values[0]:4d} | "
            f"Val {values[1]:4d} | "
            f"Test {values[2]:4d} | "
            f"Loai {values[3]:4d}"
        )

    print("\nTong anh:", len(records))

    for split in ("train", "val", "test", "exclude"):
        count = sum(
            r["split"] == split
            for r in records
        )
        print(f"{split}: {count}")

    print("\nHoan thanh luc:", datetime.now())
    print("Bao cao:", LOCKED_CSV)

    print(
        "\nLUU Y: pHash khong dam bao tim duoc "
        "tat ca anh crop/chinh sua manh."
    )


if __name__ == "__main__":
    main()
