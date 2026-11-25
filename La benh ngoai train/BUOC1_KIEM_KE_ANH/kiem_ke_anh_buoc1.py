# -*- coding: utf-8 -*-
"""BUOC 1 - Kiem ke anh 18 cay / 97 nhan. READ ONLY. Python 3 stdlib only.

Do NOT treat source folder labels as verified diagnoses.
Do NOT treat old images absent from manifest as independent test images.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PureWindowsPath

EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.jfif'}
CROP_FOLDERS = {
    'Tomato': ['Ca chua', 'Cà chua'],
    'Coffee': ['Ca phe', 'Cà phê'],
    'Rubber': ['Cao su'],
    'Lemon': ['Chanh'],
    'Tea': ['Che', 'Chè'],
    'Banana': ['Chuoi', 'Chuối'],
    'Cashew': ['Dieu', 'Điều'],
    'Papaya': ['du du', 'Du du', 'Đu đủ'],
    'Potato': ['khoai tay', 'Khoai tay', 'Khoai tây'],
    'Rice': ['Lua nuoc', 'Lúa nước'],
    'Sugarcane': ['Mia', 'Mía'],
    'Jackfruit': ['Mit', 'Mít'],
    'Corn': ['Ngo', 'Ngô'],
    'Guava': ['Oi', 'Ổi'],
    'Chili': ['Ot cay', 'Ớt cay'],
    'Pepper__bell': ['Ot chuong', 'Ớt chuông'],
    'Durian': ['Sau rieng', 'Sầu riêng'],
    'Mango': ['Xoai', 'Xoài'],
}
CROPS_VI = {
    'Banana': 'Chuối', 'Cashew': 'Điều', 'Chili': 'Ớt cay', 'Coffee': 'Cà phê',
    'Corn': 'Ngô', 'Durian': 'Sầu riêng', 'Guava': 'Ổi', 'Jackfruit': 'Mít',
    'Lemon': 'Chanh', 'Mango': 'Xoài', 'Papaya': 'Đu đủ',
    'Pepper__bell': 'Ớt chuông', 'Potato': 'Khoai tây', 'Rice': 'Lúa',
    'Rubber': 'Cao su', 'Sugarcane': 'Mía', 'Tea': 'Chè', 'Tomato': 'Cà chua',
}
# The class identifiers are copied exactly from the project's 97-class manifest.
# Label translations and common cause groups are DESCRIPTIVE ONLY, NOT diagnosis.
# If the class identifier itself is broad/ambiguous, its etiology remains UNKNOWN.
TRANSLATIONS = {
'Banana___Cordana': ('Đốm lá Cordana', 'Nấm'),
'Banana___Sigatoka': ('Bệnh đốm lá Sigatoka', 'Nấm'),
'Banana___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Cashew___Fungal_infection': ('Nhiễm nấm trên lá', 'Nấm (chưa rõ loài)'),
'Cashew___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Chili___Bacterial_spot': ('Đốm lá do vi khuẩn', 'Vi khuẩn'),
'Chili___Cercospora_leaf_spot': ('Đốm lá Cercospora', 'Nấm'),
'Chili___Curl_virus': ('Xoăn lá do virus (theo nhãn)', 'Virus (chưa xác định cụ thể)'),
'Chili___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Coffee___Phoma': ('Đốm/cháy lá Phoma', 'Nấm'),
'Coffee___Rust': ('Gỉ sắt lá cà phê', 'Nấm'),
'Coffee___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Corn___Blight': ('Bệnh cháy lá ngô', 'Tác nhân gây bệnh chưa xác định từ nhãn chung'),
'Corn___Common_rust': ('Gỉ sắt thường trên ngô', 'Nấm'),
'Corn___Gray_leaf_spot': ('Đốm lá xám', 'Nấm'),
'Corn___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Durian___Algal': ('Đốm lá do tảo', 'Tảo'),
'Durian___Blight': ('Bệnh cháy lá (nhãn chung)', 'Chưa rõ tác nhân'),
'Durian___Colletotrichum': ('Thán thư do Colletotrichum', 'Nấm'),
'Durian___Phomopsis': ('Bệnh lá liên quan Phomopsis', 'Nấm'),
'Durian___Rhizoctonia': ('Bệnh lá liên quan Rhizoctonia', 'Nấm'),
'Durian___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Guava___Anthracnose': ('Bệnh thán thư', 'Nấm'),
'Guava___Canker': ('Bệnh loét (nhãn chung)', 'Chưa rõ tác nhân'),
'Guava___Dot': ('Đốm trên lá (nhãn Dot chưa rõ)', 'Chưa rõ tác nhân'),
'Guava___Rust': ('Bệnh gỉ sắt', 'Nấm (theo tên bệnh, cần đối chiếu nguồn)'),
'Guava___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Jackfruit___Algal': ('Đốm lá do tảo', 'Tảo'),
'Jackfruit___Anthracnose': ('Bệnh thán thư', 'Nấm'),
'Jackfruit___Leaf_spot': ('Bệnh đốm lá (nhãn chung)', 'Chưa rõ tác nhân'),
'Jackfruit___Phyllosticta': ('Đốm lá Phyllosticta', 'Nấm'),
'Jackfruit___Sooty_mould': ('Nấm bồ hóng', 'Nấm bồ hóng; thường liên quan dịch ngọt từ côn trùng'),
'Jackfruit___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Lemon___Anthracnose': ('Bệnh thán thư', 'Nấm'),
'Lemon___Bacterial_blight': ('Cháy lá do vi khuẩn', 'Vi khuẩn'),
'Lemon___Citrus_canker': ('Loét cây có múi', 'Vi khuẩn'),
'Lemon___Curl_virus': ('Xoăn lá do virus (theo nhãn)', 'Virus (cần xác minh cụ thể)'),
'Lemon___Deficiency_leaf': ('Lá có dấu hiệu thiếu dinh dưỡng', 'Thiếu dinh dưỡng (chưa rõ nguyên tố)'),
'Lemon___Dry_leaf': ('Lá khô', 'Triệu chứng; chưa rõ nguyên nhân'),
'Lemon___Sooty_mould': ('Nấm bồ hóng', 'Nấm bồ hóng; có thể liên quan côn trùng chích hút'),
'Lemon___Spider_mites': ('Nhện hại lá', 'Nhện hại'),
'Lemon___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Mango___Anthracnose': ('Bệnh thán thư', 'Nấm'),
'Mango___Bacterial_canker': ('Bệnh loét do vi khuẩn', 'Vi khuẩn'),
'Mango___Cutting_weevil': ('Tổn thương do bọ vòi voi', 'Côn trùng'),
'Mango___Die_back': ('Khô chết cành/ngọn', 'Triệu chứng; nhiều nguyên nhân có thể'),
'Mango___Gall_midge': ('Tổn thương do muỗi gây u sưng', 'Côn trùng'),
'Mango___Powdery_mildew': ('Bệnh phấn trắng', 'Nấm'),
'Mango___Sooty_mould': ('Nấm bồ hóng', 'Nấm bồ hóng; thường liên quan côn trùng'),
'Mango___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Papaya___Anthracnose': ('Bệnh thán thư', 'Nấm'),
'Papaya___Bacterial_spot': ('Đốm lá do vi khuẩn', 'Vi khuẩn'),
'Papaya___Curl': ('Xoăn lá (chưa rõ nguyên nhân)', 'Chưa rõ tác nhân'),
'Papaya___Mealybug': ('Rệp sáp', 'Côn trùng'),
'Papaya___Mite_damage': ('Lá bị nhện hại', 'Nhện hại'),
'Papaya___Mosaic': ('Bệnh khảm lá (nhãn chung)', 'Có thể do virus; cần xác minh'),
'Papaya___Ringspot': ('Bệnh đốm vòng đu đủ', 'Virus (thường liên quan PRSV)'),
'Papaya___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Pepper__bell___Bacterial_spot': ('Đốm lá do vi khuẩn', 'Vi khuẩn'),
'Pepper__bell___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Potato___Early_blight': ('Cháy lá sớm', 'Nấm Alternaria'),
'Potato___Late_blight': ('Mốc sương/cháy lá muộn', 'Sinh vật giống nấm (Phytophthora)'),
'Potato___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Rice___Bacterial_leaf_blight': ('Bạc lá do vi khuẩn', 'Vi khuẩn'),
'Rice___Brown_spot': ('Bệnh đốm nâu', 'Nấm'),
'Rice___Leaf_smut': ('Bệnh than lá', 'Nấm'),
'Rubber___Anthracnose': ('Bệnh thán thư', 'Nấm'),
'Rubber___Dry_leaf': ('Lá khô', 'Triệu chứng; chưa rõ nguyên nhân'),
'Rubber___Leaf_spot': ('Bệnh đốm lá (nhãn chung)', 'Chưa rõ tác nhân'),
'Rubber___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Sugarcane___Banded_chlorosis': ('Vàng lá dạng dải', 'Triệu chứng; chưa rõ nguyên nhân'),
'Sugarcane___Brown_rust': ('Gỉ sắt nâu', 'Nấm'),
'Sugarcane___Brown_spot': ('Đốm nâu', 'Chưa rõ tác nhân từ nhãn'),
'Sugarcane___Dried_leaves': ('Lá khô', 'Triệu chứng; chưa rõ nguyên nhân'),
'Sugarcane___Grassy_shoot': ('Bệnh chồi cỏ mía', 'Vi khuẩn không vách tế bào (phytoplasma)'),
'Sugarcane___Pokkah_boeng': ('Bệnh xoăn ngọn Pokkah boeng', 'Nấm (thường thuộc nhóm Fusarium)'),
'Sugarcane___Sett_rot': ('Bệnh thối hom', 'Chưa rõ tác nhân từ nhãn'),
'Sugarcane___Smut': ('Bệnh than mía', 'Nấm'),
'Sugarcane___Viral_disease': ('Bệnh do virus (nhãn chung)', 'Virus; chưa rõ loại'),
'Sugarcane___Yellow_leaf': ('Vàng lá mía', 'Chưa rõ tác nhân chỉ từ tên bệnh'),
'Sugarcane___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Tea___Blister_blight': ('Phồng rộp lá chè', 'Nấm'),
'Tea___Brown_blight': ('Cháy lá nâu', 'Chưa rõ tác nhân từ nhãn'),
'Tea___Red_rust': ('Gỉ đỏ trên lá chè', 'Tảo (thường là Cephaleuros)'),
'Tea___Red_spider_mite': ('Nhện đỏ hại chè', 'Nhện hại'),
'Tea___Tea_mosquito_bug': ('Bọ xít muỗi hại chè', 'Côn trùng'),
'Tea___healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
'Tomato_Bacterial_spot': ('Đốm lá do vi khuẩn', 'Vi khuẩn'),
'Tomato_Early_blight': ('Cháy lá sớm', 'Nấm Alternaria'),
'Tomato_Late_blight': ('Mốc sương/cháy lá muộn', 'Sinh vật giống nấm (Phytophthora)'),
'Tomato_Leaf_Mold': ('Mốc lá', 'Nấm'),
'Tomato_Septoria_leaf_spot': ('Đốm lá Septoria', 'Nấm'),
'Tomato_Spider_mites_Two_spotted_spider_mite': ('Nhện đỏ hai chấm', 'Nhện hại'),
'Tomato__Target_Spot': ('Đốm hình bia', 'Nấm (thường là Corynespora)'),
'Tomato__Tomato_YellowLeaf__Curl_Virus': ('Virus xoăn vàng lá cà chua', 'Virus'),
'Tomato__Tomato_mosaic_virus': ('Virus khảm cà chua', 'Virus'),
'Tomato_healthy': ('Lá khỏe mạnh', 'Không có bệnh theo nhãn'),
}
FIELD_LABELS = {
    'Bell_pepper leaf': 'Pepper__bell___healthy',
    'Corn Gray leaf spot': 'Corn___Gray_leaf_spot',
    'Corn leaf blight': 'Corn___Blight',
    'Corn rust leaf': 'Corn___Common_rust',
    'Potato leaf early blight': 'Potato___Early_blight',
    'Potato leaf late blight': 'Potato___Late_blight',
    'Tomato Early blight leaf': 'Tomato_Early_blight',
    'Tomato Septoria leaf spot': 'Tomato_Septoria_leaf_spot',
    'Tomato leaf': 'Tomato_healthy',
    'Tomato leaf bacterial spot': 'Tomato_Bacterial_spot',
    'Tomato leaf late blight': 'Tomato_Late_blight',
    'Tomato leaf mosaic virus': 'Tomato__Tomato_mosaic_virus',
    'Tomato mold leaf': 'Tomato_Leaf_Mold',
    'Tomato two spotted spider mites leaf': 'Tomato_Spider_mites_Two_spotted_spider_mite',
}
UNVERIFIED_GROUPS = {'Bell_pepper leaf spot', 'Tomato leaf yellow virus'}


def norm_path(p):
    """Normalize Windows paths so case/casing differences do not cause false 'new' photos."""
    p = (p or '').strip().strip('"').replace('/', '\\')
    p = re.sub(r'\\+', r'\\', p)
    return p.casefold()


def get_crop(label):
    if label.startswith('Pepper__bell___'):
        return 'Pepper__bell'
    if label.startswith('Tomato_'):
        return 'Tomato'
    return label.split('___', 1)[0] if '___' in label else ''


def read_rows(csv_file):
    with open(csv_file, 'r', newline='', encoding='utf-8-sig') as f:
        yield from csv.DictReader(f)


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def first_existing(*paths):
    for p in paths:
        if p and p.is_file():
            return p
    return None


def scan_images(folder, origin, crop_hint, collected, errors):
    """Scan filenames only; no image decoding and no destructive operations."""
    n = 0
    def onerror(e):
        errors.append(f'Không đọc được: {getattr(e,"filename",folder)}: {e}')
    for root, dirs, files in os.walk(folder, onerror=onerror):
        dirs[:] = [d for d in dirs if d.lower() not in {'__pycache__', '.git', '$recycle.bin', 'system volume information'}]
        for name in files:
            if Path(name).suffix.lower() not in EXTS:
                continue
            p = Path(root) / name
            key = norm_path(str(p))
            # One physical path is counted once even if scanned twice.
            if key in collected:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            subfolder = ''
            try:
                relparts = p.relative_to(folder).parts
                subfolder = ' / '.join(relparts[:-1])
            except ValueError:
                pass
            collected[key] = {
                'path': str(p), 'origin_folder': origin,
                'crop_hint': crop_hint, 'disease_folder_hint': subfolder,
                'size_bytes': size,
            }
            n += 1
    return n


def determine_metadata(base_dir, project, manifest_arg, locked_arg):
    common = project / 'V10'
    manifest = first_existing(
        manifest_arg,
        common / 'dataset_manifest_SPLIT_DU_KIEN(1).csv',
        project / 'dataset_manifest_SPLIT_DU_KIEN(1).csv',
        project / 'V9' / 'dataset_manifest_SPLIT_DU_KIEN(1).csv',
        project / 'V9_AUDIT_TRAIN' / 'dataset_manifest_SPLIT_DU_KIEN(1).csv',
        base_dir / 'dataset_97_lop' / 'dataset_manifest_SPLIT_DU_KIEN(1).csv',
    )
    locked = first_existing(
        locked_arg,
        common / 'EXTERNAL_SPLIT_LOCKED.csv',
        project / 'EXTERNAL_SPLIT_LOCKED.csv',
        project / 'V9' / 'EXTERNAL_SPLIT_LOCKED.csv',
    )
    return manifest, locked


def main(argv=None):
    pa = argparse.ArgumentParser(description='Bước 1 - Kiểm kê ảnh ngoài Train, 18 cây, 97 lớp (đọc dữ liệu, không chỉnh sửa)')
    pa.add_argument('--project', default=r'D:\DoanHMvUD\La benh ngoai train', help='Thư mục có 18 thư mục cây')
    pa.add_argument('--data-root', default=r'D:\DoanHMvUD', help='Thư mục gốc chứa PlantVillage, external_field_pool_v2')
    pa.add_argument('--manifest', default='', help='Đường dẫn manifest 97 lớp nếu nằm chỗ khác')
    pa.add_argument('--locked', default='', help='Đường dẫn EXTERNAL_SPLIT_LOCKED.csv nếu nằm chỗ khác')
    pa.add_argument('--output', default='', help='Thư mục kết quả')
    pa.add_argument('--scan-extra', action='append', default=[], help='Thư mục ảnh khác cần kiểm tra, có thể chỉ định nhiều lần')
    args = pa.parse_args(argv)
    project = Path(args.project)
    data_root = Path(args.data_root)
    output = Path(args.output) if args.output else project / 'KET_QUA_KIEM_KE_BUOC1'
    if not project.is_dir():
        print('LOI: Khong tim thay thu muc do an:', project)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, locked_path = determine_metadata(data_root, project,
        Path(args.manifest) if args.manifest else None,
        Path(args.locked) if args.locked else None)
    print('== BƯỚC 1: KIỂM KÊ ẢNH ĐỒ ÁN (CHỈ ĐỌC) ==', flush=True)
    print('Đồ án:', project, flush=True)
    print('Manifest 97 lớp:', manifest_path or '(CHƯA TÌM THẤY)', flush=True)
    print('PlantDoc khóa:', locked_path or '(CHƯA TÌM THẤY)', flush=True)
    if not manifest_path and not locked_path:
        print('CHUA DU CAN CU PHAN BIET TRAIN/VAL/TEST: hãy cung cấp --manifest, --locked.', flush=True)
    # Generate full 97-label translation independent of scan success.
    trans_rows = []
    for label, (name_vi, cause) in sorted(TRANSLATIONS.items()):
        crop = get_crop(label)
        trans_rows.append({'ma_nhan_goc': label, 'ma_cay': crop,
            'ten_cay_tieng_viet': CROPS_VI.get(crop,''),
            'ten_benh_tieng_viet': name_vi,
            'nhom_nguyen_nhan_theo_ten_nhan': cause,
            'xac_minh_benh_tu_anh': 'KHONG'})
    write_csv(output / 'DANH_MUC_97_NHAN_TIENG_VIET.csv',
        ['ma_nhan_goc','ma_cay','ten_cay_tieng_viet','ten_benh_tieng_viet',
         'nhom_nguyen_nhan_theo_ten_nhan','xac_minh_benh_tu_anh'], trans_rows)
    # Read source-of-truth split labels for project metadata.
    meta = {}
    file_counts = {}
    errors = []
    duplicate_metadata_paths = 0
    def update_meta(row, origin):
        nonlocal duplicate_metadata_paths
        raw_path = row.get('path','').strip()
        if not raw_path:
            return
        key = norm_path(raw_path)
        if key in meta:
            duplicate_metadata_paths += 1
            return
        label = row.get('class','').strip() if origin == 'MANIFEST_97' else ''
        group = row.get('group','').strip() if origin == 'PLANTDOC_LOCKED' else ''
        if origin == 'PLANTDOC_LOCKED':
            label = FIELD_LABELS.get(group, '')
        crop = (get_crop(label) if label else row.get('crop','').strip())
        split = row.get('split','').strip().lower() or 'unknown'
        meta[key] = {'path': raw_path, 'origin_metadata': origin,
            'crop': crop, 'label': label, 'group': group, 'split': split,
            'needs_review': row.get('needs_label_review', '').strip() if origin=='MANIFEST_97'
                else ('YES' if group in UNVERIFIED_GROUPS or not label else 'SOURCE_GROUP_ONLY')}
    if manifest_path:
        count = 0
        for r in read_rows(manifest_path):
            update_meta(r, 'MANIFEST_97')
            count += 1
        file_counts['manifest_97_rows'] = count
        print('Manifest:', count, 'dòng', flush=True)
    if locked_path:
        count = 0
        for r in read_rows(locked_path):
            update_meta(r, 'PLANTDOC_LOCKED')
            count += 1
        file_counts['plantdoc_locked_rows'] = count
        print('PlantDoc:', count, 'dòng', flush=True)
    # 18 project crop folders + all images in PlantVillage and PlantDoc pool.
    physical = {}
    scanned_roots = []
    missing_roots = []
    found_crop_folders = {}
    used_roots = set()
    def do_scan(folder, tag, crop):
        k = norm_path(str(folder))
        if k in used_roots:
            return
        used_roots.add(k)
        if not folder.is_dir():
            missing_roots.append(str(folder))
            return
        n = scan_images(folder, tag, crop, physical, errors)
        scanned_roots.append({'folder': str(folder), 'loai': tag, 'so_anh_tim_thay': n})
        print(f'  {tag}: {n:,} ảnh | {folder}', flush=True)

    print('Bắt đầu quét thư mục trên ổ D:', flush=True)
    for crop, names in CROP_FOLDERS.items():
        found = None
        for name in names:
            maybe = project / name
            if maybe.is_dir():
                found = maybe
                break
        if found:
            found_crop_folders[crop] = str(found)
            do_scan(found, 'THU_MUC_18_CAY', crop)
        else:
            missing_roots.append('THU_MUC_CAY_CHUA_CO: ' + crop + ' | ' + ', '.join(names))
    do_scan(data_root / 'PlantVillage', 'PLANTVILLAGE', '')
    do_scan(data_root / 'external_field_pool_v2', 'PLANTDOC', '')
    # Sometimes the external pool is inside the project folder instead of data_root.
    other_pool = project / 'external_field_pool_v2'
    if other_pool.is_dir() and norm_path(str(other_pool)) != norm_path(str(data_root / 'external_field_pool_v2')):
        do_scan(other_pool, 'PLANTDOC_KHAC', '')
    for folder in args.scan_extra:
        do_scan(Path(folder), 'THU_MUC_BO_SUNG', '')
    
    # Read V10 status if present. This identifies provisional train candidates vs quarantined.
    status10 = {}
    v10 = project / 'V10'
    candidate10 = v10 / 'KET_QUA_V10' / 'TRAIN_CANDIDATE_V10.csv'
    quarantine10 = v10 / 'KET_QUA_V10' / 'TRAIN_QUARANTINE_V10.csv'
    if not candidate10.exists():
        candidate10 = v10 / 'TRAIN_CANDIDATE_V10.csv'
    if not quarantine10.exists():
        quarantine10 = v10 / 'TRAIN_QUARANTINE_V10.csv'
    for p, status in [(candidate10, 'UNG_VIEN_V10'), (quarantine10, 'CACH_LY_V10')]:
        if p.is_file():
            for r in read_rows(p):
                status10[norm_path(r.get('path',''))] = status
        else:
            missing_roots.append('KHONG_CO_FILE_V10: ' + str(p))

    # Report metadata count separately from physically observed files.
    stats97 = defaultdict(lambda: Counter())
    stats18 = defaultdict(lambda: Counter())
    metadata_outside = 0
    physical_known_outside = 0
    physical_known_train = 0
    metadata_unmatched = 0
    path_diff = set(physical) - set(meta)
    # Small flat CSVs plus full list of outside-train images and audit records.
    outside_cols = ['duong_dan_anh','nguon_anh','loai_cay_viet','ma_loai_cay',
        'nhan_benh_goc','ten_benh_tieng_viet','nhom_nguyen_nhan_goi_y',
        'nhan_thu_muc_nguon_chua_xac_minh', 'tap_du_lieu', 'trang_thai_ngoai_train',
        'trang_thai_nhan', 'trang_thai_v10', 'ton_tai_tren_o_d']
    unlabeled_cols = ['duong_dan_anh','ten_cay_goi_y','ten_thu_muc_benh_goi_y',
        'nguon_thu_muc','ly_do_can_xac_minh']
    path_outside = output / 'DANH_SACH_ANH_NGOAI_TRAIN.csv'
    path_unlisted = output / 'ANH_CHUA_CO_TRONG_DANH_SACH_CU.csv'
    path_review = output / 'ANH_CO_NHAN_CAN_RA_SOAT.csv'
    by_origin = Counter(); outside_status = Counter(); missing_by_split = Counter()
    crop_folder_image_counts = Counter(p['crop_hint'] for p in physical.values() if p['origin_folder'] == 'THU_MUC_18_CAY')
    crop_folder_subgroup_counts = defaultdict(set)
    for p in physical.values():
        if p['origin_folder'] == 'THU_MUC_18_CAY':
            crop_folder_subgroup_counts[p['crop_hint']].add(p['disease_folder_hint'] or '(không có thư mục bệnh)')
    with open(path_outside,'w',newline='',encoding='utf-8-sig') as fo, \
         open(path_unlisted,'w',newline='',encoding='utf-8-sig') as fu, \
         open(path_review,'w',newline='',encoding='utf-8-sig') as fr:
        wo = csv.DictWriter(fo, fieldnames=outside_cols); wo.writeheader()
        wu = csv.DictWriter(fu, fieldnames=unlabeled_cols); wu.writeheader()
        wr = csv.DictWriter(fr, fieldnames=outside_cols+['ly_do_ra_soat']); wr.writeheader()

        def write_external(m, p, status, outside_type, is_present):
            crop = m.get('crop','') or p.get('crop_hint','')
            label = m.get('label','')
            vi, cause = TRANSLATIONS.get(label, ('Chưa có nhãn bệnh đã ánh xạ','Chưa xác định'))
            folder = m.get('group','') or p.get('disease_folder_hint','')
            review = m.get('needs_review','')
            if m.get('origin_metadata','') == 'PLANTDOC_LOCKED':
                label_source = 'THEO_NHOM_NGUON_CHUA_XAC_MINH'
            elif label:
                label_source = 'NHAN_DATASET_CAN_DUOC_XAC_MINH'
            else:
                label_source = 'CHUA_CO_NHAN_BENH'
            row = {
                'duong_dan_anh': m.get('path') or p.get('path',''),
                'nguon_anh': m.get('origin_metadata') or p.get('origin_folder',''),
                'loai_cay_viet': CROPS_VI.get(crop,'Chưa xác định'),
                'ma_loai_cay': crop,
                'nhan_benh_goc': label,
                'ten_benh_tieng_viet': vi if label else 'Chưa xác minh',
                'nhom_nguyen_nhan_goi_y': cause if label else 'Chưa xác định',
                'nhan_thu_muc_nguon_chua_xac_minh': folder,
                'tap_du_lieu': m.get('split') or 'KHONG_CO_TRONG_METADATA',
                'trang_thai_ngoai_train': outside_type,
                'trang_thai_nhan': label_source,
                'trang_thai_v10': status,
                'ton_tai_tren_o_d': 'CO' if is_present else 'KHONG_TIM_THAY',
            }
            return row

        for k, m in meta.items():
            p = physical.get(k, {})
            split = m['split']
            c = m['crop'] or 'CHUA_XAC_DINH_CAY'
            label = m['label'] or 'CHUA_XAC_DINH_BENH'
            stats18[c]['metadata_'+split] += 1
            stats97[label]['metadata_'+split] += 1
            is_present = k in physical
            if is_present:
                stats18[c]['thuc_te_duoc_quet'] += 1
                stats97[label]['thuc_te_duoc_quet'] += 1
                if split == 'train':
                    physical_known_train += 1
                else:
                    physical_known_outside += 1
            else:
                metadata_unmatched += 1
                missing_by_split[split] += 1
            if m['origin_metadata'] == 'MANIFEST_97':
                stats97[label]['train_v10_ung_vien'] += int(split == 'train' and status10.get(k) == 'UNG_VIEN_V10')
                stats97[label]['train_v10_cach_ly'] += int(split == 'train' and status10.get(k) == 'CACH_LY_V10')
            if split != 'train':
                metadata_outside += 1
                outside_status[split] += 1
                wo.writerow(write_external(m, p, status10.get(k, ''), 'CO_TRONG_METADATA_NHUNG_KHONG_PHAI_TRAIN', is_present))
            reason = []
            if m['needs_review'] in {'YES','SOURCE_GROUP_ONLY'}:
                reason.append('NHAN_NGUON_CHUA_DUOC_XAC_MINH')
            if not m['label']:
                reason.append('KHONG_DU_NHAN_BENH')
            if status10.get(k) == 'CACH_LY_V10':
                reason.append('DA_BI_CACH_LY_V10')
            if not is_present:
                reason.append('KHONG_THAY_FILE_KHI_QUET')
            if reason and (status10.get(k)=='CACH_LY_V10' or not m['label'] or (split != 'train' and not is_present)):
                rr = write_external(m, p, status10.get(k, ''), 'THEO_METADATA', is_present)
                rr['ly_do_ra_soat'] = '; '.join(reason)
                wr.writerow(rr)

        for k in sorted(path_diff):
            p = physical[k]
            c = p['crop_hint'] or 'CHUA_XAC_DINH_CAY'
            stats18[c]['anh_ngoai_danh_sach_cu'] += 1
            if c in CROPS_VI:
                stats18[c]['anh_ngoai_danh_sach_18_cay'] += 1
            by_origin[p['origin_folder']] += 1
            outside_status['KHONG_CO_TRONG_METADATA'] += 1
            mo = {'path': p['path'], 'crop': p['crop_hint'], 'label': '', 'group': p['disease_folder_hint'], 'split': 'KHONG_CO_TRONG_METADATA'}
            wo.writerow(write_external(mo, p, '', 'CHUA_CO_TRONG_METADATA', True))
            wu.writerow({
                'duong_dan_anh': p['path'],
                'ten_cay_goi_y': CROPS_VI.get(p['crop_hint'],'Chưa xác định'),
                'ten_thu_muc_benh_goi_y': p['disease_folder_hint'],
                'nguon_thu_muc': p['origin_folder'],
                'ly_do_can_xac_minh': 'Không có trong danh sách dữ liệu cũ; tên thư mục không phải chẩn đoán; chưa chứng minh độc lập',
            })
    
    # Distinguish physical observations from metadata counts in ALL summary fields.
    by_folder_stats = Counter()
    for p in physical.values():
        by_folder_stats[(p['crop_hint'] or '(không có cây gợi ý)',
                         p['disease_folder_hint'] or '(không có nhóm bệnh gợi ý)',
                         p['origin_folder'])] += 1
    write_csv(output/'THONG_KE_THU_MUC_BENH_NGUON.csv',
        ['ma_cay_goi_y','ten_cay_viet','ten_thu_muc_benh_goi_y','nguon_thu_muc','so_anh_thuc_te'],
        ({'ma_cay_goi_y': c, 'ten_cay_viet': CROPS_VI.get(c, ''),
          'ten_thu_muc_benh_goi_y':hint,'nguon_thu_muc': ori,'so_anh_thuc_te': n}
         for (c,hint,ori),n in sorted(by_folder_stats.items())))
    write_csv(output/'THONG_KE_18_LOAI_CAY.csv',
        ['ma_cay','ten_cay_tieng_viet','duong_dan_thu_muc_cay',
         'metadata_train','metadata_val','metadata_test','metadata_exclude_review',
         'metadata_ngoai_train','anh_thuc_te_duoc_quet_trong_metadata',
         'so_anh_trong_thu_muc_cay','so_nhom_thu_muc_benh_goi_y',
         'anh_ngoai_danh_sach_cu','canh_bao'],
        ({'ma_cay': c,'ten_cay_tieng_viet': CROPS_VI[c],
          'duong_dan_thu_muc_cay': found_crop_folders.get(c,''),
          'metadata_train': stats18[c]['metadata_train'],
          'metadata_val': stats18[c]['metadata_val'],
          'metadata_test': stats18[c]['metadata_test'],
          'metadata_exclude_review': stats18[c]['metadata_exclude_review'] + stats18[c]['metadata_exclude'],
          'metadata_ngoai_train': sum(n for k,n in stats18[c].items() if k.startswith('metadata_') and k!='metadata_train'),
          'anh_thuc_te_duoc_quet_trong_metadata': stats18[c]['thuc_te_duoc_quet'],
          'so_anh_trong_thu_muc_cay': crop_folder_image_counts[c],
          'so_nhom_thu_muc_benh_goi_y': len(crop_folder_subgroup_counts[c]),
          'anh_ngoai_danh_sach_cu': stats18[c]['anh_ngoai_danh_sach_cu'],
          'canh_bao': 'CHUA_CO_THU_MUC_CAY' if c not in found_crop_folders else 'TEN_THU_MUC_BENH_CHUA_DUOC_XAC_MINH'}
         for c in CROPS_VI))
    write_csv(output/'THONG_KE_97_NHAN_BENH.csv',
        ['nhan_tieng_anh','ten_cay_tieng_viet','benh_tieng_viet','nguyen_nhan_goi_y',
         'metadata_train','metadata_val','metadata_test','metadata_exclude_review',
         'metadata_tong','anh_thuc_te_duoc_quet_trong_metadata',
         'train_v10_ung_vien','train_v10_cach_ly'],
        ({'nhan_tieng_anh': label, 'ten_cay_tieng_viet': CROPS_VI.get(get_crop(label),''),
          'benh_tieng_viet': TRANSLATIONS[label][0], 'nguyen_nhan_goi_y': TRANSLATIONS[label][1],
          'metadata_train': stats97[label]['metadata_train'],
          'metadata_val': stats97[label]['metadata_val'],
          'metadata_test': stats97[label]['metadata_test'],
          'metadata_exclude_review': stats97[label]['metadata_exclude_review'] + stats97[label]['metadata_exclude'],
          'metadata_tong': sum(n for k,n in stats97[label].items() if k.startswith('metadata_')),
          'anh_thuc_te_duoc_quet_trong_metadata': stats97[label]['thuc_te_duoc_quet'],
          'train_v10_ung_vien': stats97[label]['train_v10_ung_vien'],
          'train_v10_cach_ly': stats97[label]['train_v10_cach_ly']}
         for label in sorted(TRANSLATIONS)))
    # Validate coverage of model class list against source 97 classes.
    declared_classes = {m['label'] for m in meta.values() if m['origin_metadata']=='MANIFEST_97'}
    missing_trans = sorted(declared_classes - set(TRANSLATIONS))
    unused_trans = sorted(set(TRANSLATIONS) - declared_classes) if manifest_path else []
    metadata_split_counts = Counter(m['split'] for m in meta.values())
    inventory = {
        'che_do': 'READ_ONLY_AUDIT_NOT_AI_DIAGNOSIS',
        'thoi_diem_chay': datetime.now().isoformat(timespec='seconds'),
        'so_thu_muc_loai_cay_tim_thay': len(found_crop_folders),
        'so_thu_muc_loai_cay_mong_doi': 18,
        'so_nhan_benh_danh_muc': len(TRANSLATIONS),
        'metadata_97': str(manifest_path) if manifest_path else None,
        'metadata_plantdoc': str(locked_path) if locked_path else None,
        'so_dong_du_lieu_goc': file_counts,
        'so_dong_metadata_khong_trung_duong_dan': len(meta),
        'so_duong_dan_trung_trong_metadata': duplicate_metadata_paths,
        'theo_nhan_chia_trong_metadata': dict(metadata_split_counts),
        'so_anh_ngoai_train_theo_metadata': metadata_outside,
        'so_anh_vat_ly_quyet_dinh_ngoai_train_bang_file_chia': physical_known_outside,
        'so_anh_vat_ly_quyet_dinh_train_bang_file_chia': physical_known_train,
        'so_anh_vat_ly_chua_xac_dinh_split': len(path_diff),
        'so_anh_trong_18_thu_muc_cay': sum(crop_folder_image_counts.values()),
        'so_anh_tim_duoc_ngoai_danh_sach_metadata': len(path_diff),
        'so_anh_duong_dan_ngoai_train_can_ra_soat': metadata_outside + len(path_diff),
        'so_anh_thuc_te_quet_duoc': len(physical),
        'so_anh_metadata_khong_xuat_hien_trong_cac_goc_da_quet': metadata_unmatched,
        'so_anh_metadata_khong_xuat_hien_theo_split': dict(missing_by_split),
        'so_anh_ngoai_metadata_theo_nguon': dict(by_origin),
        'cac_goc_da_quet': scanned_roots,
        'goc_thieu': missing_roots,
        'loi_truy_cap': errors[:100],
        'danh_muc_97_chua_duoc_dich': missing_trans,
        'danh_muc_97_khong_nam_trong_manifest': unused_trans,
        'canh_bao': [
            'Đếm ảnh theo đường dẫn, chưa kiểm tra trùng nội dung SHA256/pHash. Ảnh trùng đổi tên có thể được tính nhiều lần.',
            'Ảnh không nằm trong manifest KHÔNG đồng nghĩa là ảnh mới hoàn toàn độc lập hoặc có nhãn đúng.',
            'Nhãn trong dữ liệu chỉ là nhãn nguồn; KHÔNG phải kết luận bệnh từ pixels.',
            'Thư mục test/train trong đường dẫn KHÔNG được dùng để quyết định split: chỉ tin CSV đã khóa.',
            'Các ảnh chưa có trong CSV cũ có thể đã được huấn luyện ở một nơi khác; KHÔNG được xem là ngoài Train chắc chắn.',
            'Các trường metadata đếm toàn bộ CSV, kể cả ảnh không có trong gốc đã quét.',
            'Dữ liệu nguồn chưa được chuyên gia xác minh, không dùng các số này làm Accuracy.',
            'Split=train trong manifest không chứng minh ảnh chắc chắn đã được nạp trong từng lần huấn luyện lấy mẫu V2/V3/V11.',
        ],
    }
    (output/'TONG_KET_BUOC1.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    summary_lines = [
        'BƯỚC 1 - KIỂM KÊ ẢNH CÂY/BỆNH. ĐÂY KHÔNG PHẢI PHẦN MỀM NHẬN DIỆN BỆNH.','',
        f'Gốc dự án: {project}',
        f'Manifest 97 lớp: {manifest_path or "CHƯA TÌM THẤY"}',
        f'File PlantDoc khóa: {locked_path or "CHƯA TÌM THẤY"}',
        f'Số thư mục cây tìm thấy: {len(found_crop_folders)}/18',
        f'Số ảnh thực tế quét được (không trùng đường dẫn): {len(physical):,}',
        f'Số ảnh trong 18 thư mục cây riêng: {sum(crop_folder_image_counts.values()):,}',
        f'Số dòng metadata duy nhất: {len(meta):,}',
        f'Số ảnh theo metadata thuộc Train: {metadata_split_counts["train"]:,}',
        f'Số ảnh theo metadata ngoài Train: {metadata_outside:,}',
        f'Số ảnh thực tế tìm thấy thuộc nhóm ngoài Train theo CSV: {physical_known_outside:,}',
        f'Số ảnh thực tế tìm thấy thuộc nhóm Train theo CSV: {physical_known_train:,}',
        f'Số ảnh vật lý ngoài danh sách CSV cũ: {len(path_diff):,}',
        f'TỔNG đường dẫn ngoài Train theo CSV + chưa có trong CSV cũ: {metadata_outside + len(path_diff):,}',
        f'Số metadata chưa thấy khi quét các gốc chỉ định: {metadata_unmatched:,}',
        '', 'CẢNH BÁO: số ảnh ngoài Train bao gồm ảnh Validation/Test cũ, KHÔNG phải tập Test độc lập mới.',
        'CẢNH BÁO: bệnh được liệt kê theo nhãn nguồn/thư mục, CHƯA được AI phân tích hình ảnh hay chuyên gia xác minh.',
        'CẢNH BÁO: số dòng CSV metadata và số ảnh thực tế là hai chỉ số khác nhau.',
        'CẢNH BÁO: split=train trong CSV không chứng minh ảnh đã được dùng trong một epoch huấn luyện cụ thể.',
        '', 'KẾT QUẢ CHI TIẾT:',
        '- TONG_KET_BUOC1.json',
        '- THONG_KE_18_LOAI_CAY.csv',
        '- THONG_KE_97_NHAN_BENH.csv',
        '- THONG_KE_THU_MUC_BENH_NGUON.csv',
        '- DANH_SACH_ANH_NGOAI_TRAIN.csv',
        '- ANH_CHUA_CO_TRONG_DANH_SACH_CU.csv',
        '- ANH_CO_NHAN_CAN_RA_SOAT.csv',
        '- DANH_MUC_97_NHAN_TIENG_VIET.csv',
        '', 'GỬI CHATGPT 4 FILE:',
        '  1. TONG_KET_BUOC1.json',
        '  2. THONG_KE_18_LOAI_CAY.csv',
        '  3. THONG_KE_97_NHAN_BENH.csv',
        '  4. THONG_KE_THU_MUC_BENH_NGUON.csv',
        'Nếu có cảnh báo về metadata thiếu, gửi thêm BAO_CAO_BUOC1.txt.',
    ]
    (output/'BAO_CAO_BUOC1.txt').write_text('\n'.join(summary_lines)+'\n',encoding='utf-8-sig')
    print('\n'.join(summary_lines[:13]),flush=True)
    print('XONG. Kết quả ở:', output, flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Đã dừng theo yêu cầu. Không có ảnh gốc bị thay đổi.')
        sys.exit(130)
    except Exception as exc:
        print('LOI:', type(exc).__name__, str(exc))
        raise
