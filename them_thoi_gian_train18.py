
# -*- coding: utf-8 -*-
"""
Them dong ho, ETA epoch va ETA toan bo
vao train_18_crops_final.py.

Khong thay doi model, data hay thuat toan train.
"""

from pathlib import Path
import ast
import shutil

ROOT = Path(__file__).resolve().parent
APP = ROOT / "train_18_crops_final.py"
BACKUP = ROOT / "train_18_crops_final_truoc_eta.py"

if not APP.exists():
    raise FileNotFoundError(
        f"Khong tim thay: {APP}"
    )

code = APP.read_text(encoding="utf-8-sig")

if "# ETA_TRAIN18_V1" in code:
    print("Da them dong ho truoc do. Khong can sua nua.")
    raise SystemExit(0)


def replace_once(old, new):
    global code

    if code.count(old) != 1:
        raise ValueError(
            "Khong tim thay dung vi tri can sua:\n"
            + old[:120]
            + "\nFile goc chua bi thay doi."
        )

    code = code.replace(old, new, 1)


# 1. Thu vien do thoi gian
replace_once(
    "import random\n",
    "import random\nimport time\n"
)

replace_once(
    "from datetime import datetime\n",
    "from datetime import datetime, timedelta\n"
)


# 2. Ham dinh dang gio:phut:giay
replace_once(
    "def crop_of(label):",
    '''
# ETA_TRAIN18_V1
def format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def crop_of(label):'''
)


# 3. Bat dau tinh thoi gian chuong trinh
replace_once(
    "    args = parser.parse_args()\n",
    '''    args = parser.parse_args()

    session_started = time.perf_counter()

    print(
        "\\nBat dau luc:",
        datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
        flush=True
    )
'''
)


# 4. Luu thoi gian validation cua epoch truoc
replace_once(
    '''    criterion_val = nn.CrossEntropyLoss()

    for epoch in range(''',
    '''    criterion_val = nn.CrossEntropyLoss()

    last_val_seconds = None

    for epoch in range('''
)


# 5. Bat dau dem gio moi epoch
replace_once(
    '''        for step, (images, labels) in enumerate(
            loader, 1
        ):''',
    '''        epoch_started = time.perf_counter()
        train_started = time.perf_counter()

        for step, (images, labels) in enumerate(
            loader, 1
        ):'''
)


# 6. Them ETA vao thong bao tien do
replace_once(
    '''                print(
                    f"Train {step}/{len(loader)} "
                    f"({step / len(loader):.1%})",
                    flush=True
                )''',
    '''                elapsed_train = (
                    time.perf_counter() - train_started
                )

                sec_per_batch = (
                    elapsed_train / max(step, 1)
                )

                train_remaining = (
                    len(loader) - step
                ) * sec_per_batch

                # Epoch dau: uoc luong thoi gian
                # validation tu toc do batch Train.
                # Epoch sau: dung thoi gian Val thuc te.
                if last_val_seconds is None:
                    val_est = (
                        len(val_loader)
                        * sec_per_batch
                        * 0.5
                    )
                else:
                    val_est = last_val_seconds

                epochs_left = max(
                    0, args.epochs - epoch
                )

                full_epoch_est = (
                    len(loader) * sec_per_batch
                    + val_est
                )

                # Uoc tinh Test, khong bao gom
                # external_test neu co nhieu anh.
                test_est = (
                    math.ceil(len(test_rows) / 16)
                    * sec_per_batch
                    * 0.5
                )

                eta_total = (
                    train_remaining
                    + val_est
                    + epochs_left * full_epoch_est
                    + test_est
                )

                finish_time = (
                    datetime.now()
                    + timedelta(seconds=eta_total)
                )

                print(
                    f"Train {step}/{len(loader)} "
                    f"({step / len(loader):.1%})"
                )

                print(
                    "  Da chay: "
                    + format_duration(
                        time.perf_counter()
                        - session_started
                    )
                )

                print(
                    "  Con lai epoch: "
                    + format_duration(train_remaining)
                )

                print(
                    "  Con lai toan bo (uoc tinh): "
                    + format_duration(eta_total)
                )

                print(
                    "  Du kien xong: "
                    + finish_time.strftime(
                        "%H:%M:%S %d/%m/%Y"
                    ),
                    flush=True
                )'''
)


# 7. Ghi nhan thoi gian Validation thuc te
replace_once(
    '''        val_result = evaluate(
            model, val_loader, device, amp
        )

        val_acc, val_f1 = val_result[:2]''',
    '''        print(
            "\\nDang danh gia Validation...",
            flush=True
        )

        val_started = time.perf_counter()

        val_result = evaluate(
            model, val_loader, device, amp
        )

        last_val_seconds = (
            time.perf_counter() - val_started
        )

        print(
            "Thoi gian epoch: "
            + format_duration(
                time.perf_counter() - epoch_started
            )
        )

        print(
            "Thoi gian Validation: "
            + format_duration(last_val_seconds)
        )

        val_acc, val_f1 = val_result[:2]'''
)


# 8. Hien tong thoi gian khi ket thuc
replace_once(
    '''    print("\\n===== HOAN THANH =====")
    print("Best Validation Macro F1:", best_f1)''',
    '''    print("\\n===== HOAN THANH =====")

    print(
        "Tong thoi gian phien chay: "
        + format_duration(
            time.perf_counter() - session_started
        )
    )

    print(
        "Hoan thanh luc:",
        datetime.now().strftime("%H:%M:%S %d/%m/%Y")
    )

    print("Best Validation Macro F1:", best_f1)'''
)


# 9. Kiem tra cu phap truoc khi ghi file
ast.parse(code)
compile(code, str(APP), "exec")

if not BACKUP.exists():
    shutil.copy2(APP, BACKUP)

APP.write_text(code, encoding="utf-8")

print("\\n===== CAP NHAT THANH CONG =====")
print("File da cap nhat:", APP)
print("Ban sao luu:", BACKUP)
print("Da them ETA epoch, ETA toan bo va gio hoan thanh.")
