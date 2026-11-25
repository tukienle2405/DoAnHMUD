
from pathlib import Path
from PIL import Image
from collections import defaultdict
import hashlib
import csv

ROOT = Path(r"D:\DoanHMvUD")
DATA = ROOT / "PlantVillage" / "Papaya Leaf Disease Image Dataset" / "new"
OUT = ROOT / "dataset_review"
OUT.mkdir(exist_ok=True)

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
records = []
errors = []

def dhash(path):
    with Image.open(path) as img:
        im = img.convert("L").resize((9, 8))
        pixels = list(im.tobytes())

    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (
                pixels[y * 9 + x] > pixels[y * 9 + x + 1]
            )
    return value

def save_csv(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

for folder in sorted(DATA.iterdir()):
    if not folder.is_dir():
        continue

    for path in folder.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXT:
            continue

        try:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            visual = dhash(path)
            records.append((folder.name, path, sha, visual))
        except Exception as e:
            errors.append([str(path), str(e)])

print("Da doc:", len(records), "anh")

# 1. Anh trung chinh xac giua cac lop
by_sha = defaultdict(list)
by_name = defaultdict(list)

for cls, path, sha, visual in records:
    by_sha[sha].append((cls, path))
    by_name[path.name.lower()].append((cls, path))

exact = []
for sha, items in by_sha.items():
    if len({cls for cls, _ in items}) > 1:
        for cls, path in items:
            exact.append([sha, cls, str(path)])

# 2. Trung ten file giua cac lop
same_names = []
for name, items in by_name.items():
    classes = sorted({cls for cls, _ in items})
    if len(classes) > 1:
        same_names.append([name, " | ".join(classes)])

# 3. Anh gan trung giua cac lop (dHash)
near = []
limit = 3000

for i in range(len(records)):
    cls1, path1, sha1, h1 = records[i]

    for cls2, path2, sha2, h2 in records[i+1:]:
        if cls1 == cls2 or sha1 == sha2:
            continue

        distance = (h1 ^ h2).bit_count()

        if distance <= 3:
            near.append([
                cls1, cls2, distance,
                str(path1), str(path2)
            ])

            if len(near) >= limit:
                break

    if len(near) >= limit:
        break

save_csv(
    "papaya_trung_chinh_xac.csv",
    ["sha256", "class", "path"],
    exact
)

save_csv(
    "papaya_trung_ten.csv",
    ["filename", "classes"],
    same_names
)

save_csv(
    "papaya_gan_trung.csv",
    ["class1", "class2", "distance", "path1", "path2"],
    near
)

save_csv(
    "papaya_anh_loi.csv",
    ["path", "error"],
    errors
)

print("\n===== KET QUA =====")
print("Anh trung chinh xac (so dong):", len(exact))
print("Ten file xuat hien o nhieu lop:", len(same_names))
print("Cap anh nghi gan trung:", len(near))
print("Anh loi:", len(errors))
print("Da luu bao cao tai:", OUT)
print("Luu y: Gan trung chi la nghi van, khong tu dong xoa anh.")
