
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import hashlib
import csv

# =========================
# 1. CAU HINH
# =========================

ROOT = Path(r"D:\DoanHMvUD")

DATA = (
    ROOT / "PlantVillage"
    / "Papaya Leaf Disease Image Dataset"
    / "new"
)

OUTPUT = ROOT / "dataset_review" / "papaya_clean"
OUTPUT.mkdir(parents=True, exist_ok=True)

EXT = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".tif", ".tiff"
}

# Chi la nguong tim anh nghi gan trung.
# Khong phai nguong chung minh anh trung.
NEAR_THRESHOLD = 3


# =========================
# 2. CAC HAM HO TRO
# =========================

def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def dhash_file(path):
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize(
            (9, 8),
            Image.Resampling.LANCZOS
        )

        pixels = img.tobytes()

    value = 0

    for y in range(8):
        for x in range(8):
            a = pixels[y * 9 + x]
            b = pixels[y * 9 + x + 1]

            value = (value << 1) | (a > b)

    return value


def suspicious_filename(name):
    name = name.lower()

    keywords = (
        "_increased_",
        "_decreased_",
        "_aug_",
        "_rotated",
        " - copy",
        "_copy"
    )

    return any(word in name for word in keywords)


def save_csv(filename, header, rows):
    path = OUTPUT / filename

    with open(
        path, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# =========================
# 3. DOC TAT CA ANH GOC
# =========================

if not DATA.is_dir():
    raise FileNotFoundError(
        f"Khong tim thay duong dan: {DATA}"
    )

records = []
errors = []

for folder in sorted(DATA.iterdir()):

    if not folder.is_dir():
        continue

    for path in sorted(folder.rglob("*")):

        if not path.is_file():
            continue

        if path.suffix.lower() not in EXT:
            continue

        try:
            record = {
                "class": folder.name,
                "path": str(path),
                "sha": sha256_file(path),
                "dhash": dhash_file(path)
            }

            records.append(record)

        except Exception as e:
            errors.append([str(path), str(e)])

print("Da doc:", len(records), "anh")


# =========================
# 4. TIM ANH TRUNG CHINH XAC
# =========================

by_sha = defaultdict(list)

for record in records:
    by_sha[record["sha"]].append(record)

conflicts = set()
duplicates = set()
conflict_groups = []

for sha, items in by_sha.items():

    classes = {x["class"] for x in items}

    # Cung mot anh nhung nhieu nhan khac nhau
    if len(classes) > 1:

        for item in items:
            conflicts.add(item["path"])

            conflict_groups.append([
                sha,
                item["class"],
                item["path"]
            ])

    # Trung trong cung mot lop: giu mot ban
    elif len(items) > 1:

        items = sorted(
            items, key=lambda x: x["path"]
        )

        for item in items[1:]:
            duplicates.add(item["path"])


# =========================
# 5. TIM ANH GAN TRUNG
# =========================

near_paths = set()
near_pairs = []

# Chi kiem tra cac file chua bi cach ly
remaining = [
    r for r in records
    if r["path"] not in conflicts
    and r["path"] not in duplicates
]

for i in range(len(remaining)):

    a = remaining[i]

    for j in range(i + 1, len(remaining)):

        b = remaining[j]

        # Chi canh bao neu khac nhan
        if a["class"] == b["class"]:
            continue

        distance = (
            a["dhash"] ^ b["dhash"]
        ).bit_count()

        if distance <= NEAR_THRESHOLD:

            near_paths.add(a["path"])
            near_paths.add(b["path"])

            near_pairs.append([
                a["class"],
                b["class"],
                distance,
                a["path"],
                b["path"]
            ])


# =========================
# 6. PHAN LOAI TUNG ANH
# =========================

manifest = []
candidates = []
quarantine = []
counts = Counter()

for record in records:

    path = record["path"]
    cls = record["class"]

    if path in conflicts:

        status = "CACH_LY_KHAC_NHAN"
        reason = "Trung SHA256 giua nhieu lop"

    elif path in duplicates:

        status = "BO_BAN_SAO_CUNG_LOP"
        reason = "Trung SHA256 trong cung lop"

    elif path in near_paths:

        status = "CAN_XEM_GAN_TRUNG"
        reason = "dHash gan giong anh khac lop"

    elif suspicious_filename(Path(path).name):

        status = "CAN_XAC_MINH_NGUON"
        reason = "Ten file nghi da augmentation"

    else:

        status = "DE_XUAT_SAU_KIEM_TRA"
        reason = "Chua phat hien van de theo quy tac"

    counts[status] += 1

    manifest.append([
        path,
        cls,
        record["sha"],
        status,
        reason
    ])

    if status == "DE_XUAT_SAU_KIEM_TRA":

        candidates.append([
            path,
            cls
        ])

    else:

        quarantine.append([
            path,
            cls,
            status,
            reason
        ])


# =========================
# 7. XUAT BAO CAO
# =========================

save_csv(
    "tat_ca_anh_phan_loai.csv",
    ["path", "class", "sha256", "status", "reason"],
    manifest
)

save_csv(
    "anh_de_xuat.csv",
    ["path", "class"],
    candidates
)

save_csv(
    "anh_tam_cach_ly.csv",
    ["path", "class", "status", "reason"],
    quarantine
)

save_csv(
    "nhom_trung_khac_nhan.csv",
    ["sha256", "class", "path"],
    conflict_groups
)

save_csv(
    "cap_anh_gan_trung.csv",
    ["class1", "class2", "distance", "path1", "path2"],
    near_pairs
)

save_csv(
    "anh_loi.csv",
    ["path", "error"],
    errors
)

report = [
    "BAO CAO LAM SACH PAPAYA",
    "=" * 45,
    f"Tong anh doc thanh cong: {len(records)}",
    f"So anh loi: {len(errors)}",
    "",
    "KET QUA PHAN LOAI:"
]

for status, count in sorted(counts.items()):
    report.append(f"{status}: {count}")

report.extend([
    "",
    f"So cap anh nghi gan trung: {len(near_pairs)}",
    f"So anh de xuat: {len(candidates)}",
    "",
    "CHU Y:",
    "Chua xac nhan nhan benh bang chuyen gia.",
    "Chua kiem tra day du anh gan trung cung lop.",
    "Chua chia Train/Validation/Test.",
    "Khong xoa hay di chuyen anh goc."
])

text = "\n".join(report)

(OUTPUT / "bao_cao_lam_sach.txt").write_text(
    text, encoding="utf-8"
)

print("\n" + text)
print("\nDa luu tai:", OUTPUT)
