
from pathlib import Path
import importlib.util
import csv

ROOT = Path(r"D:\DoanHMvUD")

SCRIPT = (
    ROOT / "La benh ngoai train"
    / "danh_gia_plantdoc_1194_co_thoi_gian.py"
)

LOCKED = (
    ROOT / "external_field_pool_v2"
    / "field_adaptation_v1"
    / "EXTERNAL_SPLIT_LOCKED.csv"
)

# Tự tìm checkpoint ứng viên mới nhất
candidates = list(
    (ROOT / "training_runs").glob(
        "crop18_field_pilot_*/best_field_candidate.pth"
    )
)

if not candidates:
    raise FileNotFoundError(
        "Khong tim thay best_field_candidate.pth"
    )

CHECKPOINT = max(
    candidates,
    key=lambda p: p.stat().st_mtime
)

if not SCRIPT.is_file():
    raise FileNotFoundError(SCRIPT)

if not LOCKED.is_file():
    raise FileNotFoundError(LOCKED)

with LOCKED.open(
    encoding="utf-8-sig", newline=""
) as f:
    test_paths = {
        row["path"].lower()
        for row in csv.DictReader(f)
        if row["split"] == "test"
        and row["reason"] == "OK"
    }

print("Checkpoint 18 cay moi:", CHECKPOINT)
print("So anh Test duoc khoa:", len(test_paths))

if len(test_paths) != 238:
    raise ValueError(
        "So anh Test thay doi. "
        "Can kiem tra lai truoc khi danh gia."
    )

# Nạp chương trình đánh giá đã hoạt động
spec = importlib.util.spec_from_file_location(
    "plantdoc_evaluator",
    SCRIPT
)

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Dùng mô hình 18 cây mới
module.MODEL_18 = CHECKPOINT

# Không ghi đè báo cáo trước
module.OUTPUT = (
    ROOT / "external_field_pool_v2"
    / "field_adaptation_v1"
    / "KET_QUA_KET_HOP_18_97"
)

original_read_images = module.read_images


def read_test_only():
    all_images = original_read_images()

    selected = [
        record for record in all_images
        if str(record["path"]).lower() in test_paths
    ]

    if len(selected) != len(test_paths):
        raise ValueError(
            f"Chi tim thay {len(selected)}/"
            f"{len(test_paths)} anh Test"
        )

    print("CHI DANH GIA ANH TEST:", len(selected))
    return selected


module.read_images = read_test_only

# Chạy mô hình 18 cây + mô hình 97 lớp
module.main()
