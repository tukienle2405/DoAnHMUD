
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import csv

ROOT = Path(r"D:\DoanHMvUD")

BASE = ROOT / "dataset_review" / "dataset_97_lop"
INPUT = BASE / "dataset_manifest_SO_BO.csv"
OUT = BASE / "kiem_tra_trung"

OUT.mkdir(parents=True, exist_ok=True)

REAL_SPLITS = {"train", "val", "test"}


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)

    return h.hexdigest()


def save_csv(filename, header, rows):
    with open(
        OUT / filename,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


if not INPUT.is_file():
    raise FileNotFoundError(INPUT)

print("Dang doc dataset manifest...")

with open(INPUT, encoding="utf-8-sig", newline="") as f:
    records = list(csv.DictReader(f))

print("Tong anh trong manifest:", len(records))

# ====================================
# 1. KIEM TRA DUONG DAN VA KICH THUOC
# ====================================

by_size = defaultdict(list)
missing = []
errors = []

for i, record in enumerate(records):
    path = Path(record["path"])

    try:
        if not path.is_file():
            missing.append([str(path), "Khong ton tai"])
            continue

        size = path.stat().st_size
        by_size[size].append(i)

    except Exception as exc:
        errors.append([str(path), str(exc)])

print("Da kiem tra duong dan.")


# ====================================
# 2. TINH SHA256 CHO FILE CO CUNG SIZE
# ====================================

by_sha = defaultdict(list)
hashed = 0

for size, indexes in by_size.items():

    if len(indexes) < 2:
        continue

    for i in indexes:
        record = records[i]
        path = Path(record["path"])

        try:
            digest = sha256_file(path)
            by_sha[digest].append(i)
            hashed += 1

        except Exception as exc:
            errors.append([str(path), str(exc)])

        if hashed % 5000 == 0:
            print("Da tinh SHA256:", hashed)

print("Da hoan thanh SHA256.")


# ====================================
# 3. PHAN LOAI NHOM ANH TRUNG
# ====================================

duplicate_rows = []
group_rows = []
quarantine_rows = []

group_counts = Counter()
affected_paths = defaultdict(set)

duplicate_groups = {
    sha: indexes
    for sha, indexes in by_sha.items()
    if len(indexes) > 1
}

for sha, indexes in duplicate_groups.items():

    items = [records[i] for i in indexes]

    labels = {r["class"] for r in items}
    splits = {r["split"] for r in items}
    fixed_splits = splits & REAL_SPLITS

    if len(labels) > 1:
        issue = "MAU_THUAN_NHAN"

    elif len(fixed_splits) > 1:
        issue = "TRUNG_QUA_NHIEU_TAP"

    elif fixed_splits and "pending" in splits:
        issue = "GIAO_PENDING_VOI_TAP_DA_CHIA"

    else:
        issue = "TRUNG_CUNG_NHAN"

    group_counts[issue] += 1

    group_rows.append([
        sha,
        issue,
        len(items),
        " | ".join(sorted(labels)),
        " | ".join(sorted(splits))
    ])

    for r in items:
        duplicate_rows.append([
            sha,
            issue,
            r["class"],
            r["split"],
            r["source"],
            r["path"]
        ])

        affected_paths[issue].add(r["path"])

        if issue != "TRUNG_CUNG_NHAN":
            quarantine_rows.append([
                r["path"],
                r["class"],
                r["split"],
                issue
            ])


# ====================================
# 4. KIEM TRA RO RI THEO GROUP_ID
# ====================================

group_splits = defaultdict(set)

for r in records:

    group_id = r["group_id"].strip()
    split = r["split"]

    if not group_id or split not in REAL_SPLITS:
        continue

    key = (r["class"], group_id)
    group_splits[key].add(split)

group_leakage = []

for (label, group_id), splits in group_splits.items():

    if len(splits) > 1:
        group_leakage.append([
            label,
            group_id,
            " | ".join(sorted(splits))
        ])


# ====================================
# 5. XUAT BAO CAO
# ====================================

save_csv(
    "nhom_anh_trung_sha.csv",
    [
        "sha256",
        "issue",
        "file_count",
        "classes",
        "splits"
    ],
    group_rows
)

save_csv(
    "chi_tiet_anh_trung.csv",
    [
        "sha256",
        "issue",
        "class",
        "split",
        "source",
        "path"
    ],
    duplicate_rows
)

save_csv(
    "anh_can_cach_ly.csv",
    ["path", "class", "split", "reason"],
    quarantine_rows
)

save_csv(
    "ro_ri_theo_group_id.csv",
    ["class", "group_id", "splits"],
    group_leakage
)

save_csv(
    "duong_dan_thieu.csv",
    ["path", "error"],
    missing
)

save_csv(
    "loi_doc_file.csv",
    ["path", "error"],
    errors
)


# ====================================
# 6. IN THONG KE
# ====================================

report = [
    "BAO CAO KIEM TRA TRUNG LAP 97 LOP",
    "=" * 50,
    f"Tong anh trong manifest: {len(records)}",
    f"So file da tinh SHA256: {hashed}",
    f"So nhom trung chinh xac: {len(duplicate_groups)}",
    f"So dong anh trong nhom trung: {len(duplicate_rows)}",
    "",
    "PHAN LOAI NHOM TRUNG:"
]

for issue, count in sorted(group_counts.items()):
    report.append(
        f"{issue}: {count} nhom, "
        f"{len(affected_paths[issue])} duong dan"
    )

report.extend([
    "",
    f"So nhom ro ri group_id: {len(group_leakage)}",
    f"So duong dan thieu: {len(missing)}",
    f"So loi doc file: {len(errors)}",
    "",
    "LUU Y:",
    "SHA256 chi phat hien trung noi dung chinh xac.",
    "Chua kiem tra anh gan trung hoac augmentation.",
    "Khong tu dong xoa hoac loai anh.",
    "Chua tao Train/Validation/Test cho anh pending."
])

report_text = "\n".join(report)

(OUT / "bao_cao_trung_97_lop.txt").write_text(
    report_text, encoding="utf-8"
)

print("\n" + report_text)
print("\nKet qua luu tai:", OUT)
