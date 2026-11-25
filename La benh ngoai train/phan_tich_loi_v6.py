# -*- coding: utf-8 -*-
"""V6: audit errors from V5 on validation, no training, no test, no model changes.

Usage: py -3 -u phan_tich_loi_v6.py
Optional: --csv PATH --lock PATH --out PATH
No external Python packages required.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import mimetypes
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = ['path', 'true_crop', 'true_disease', 'pred_crop', 'pred_disease',
          'crop_ok', 'joint_ok', 'max_blend_probability', 'model_disagreement',
          'old_crop_pred', 'v3_crop_pred']


def readcsv(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def writecsv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def accuracy(rows, field):
    return round(100 * sum(x[field] == '1' for x in rows) / len(rows), 2) if rows else None


def read_lock(lock_path, val_rows):
    if not lock_path.is_file():
        return {'lock_found': False, 'lock_path': str(lock_path),
                'note': 'Chua xac thuc lai split lock; khong duoc suy luan ve TEST'}
    locked = readcsv(lock_path)
    index = {x['path'].casefold().replace('/', '\\'): x for x in locked}
    missing, unexpected = [], []
    for row in val_rows:
        path = row['path'].casefold().replace('/', '\\')
        if path not in index:
            missing.append(row['path'])
        elif index[path]['split'] != 'val':
            unexpected.append({'path': row['path'], 'actual_split': index[path]['split']})
    return {'lock_found': True, 'lock_path': str(lock_path), 'rows': len(val_rows),
            'matched_val': len(val_rows) - len(missing) - len(unexpected),
            'not_in_lock': missing, 'not_val': unexpected,
            'ok': not missing and not unexpected}


def thumb_image(path, max_bytes=3_000_000):
    fp = Path(path)
    try:
        if not fp.is_file() or fp.stat().st_size > max_bytes:
            return ''
        mt, _ = mimetypes.guess_type(str(fp))
        if mt not in ('image/jpeg', 'image/png', 'image/webp', 'image/gif'):
            return ''
        return f'data:{mt};base64,' + base64.b64encode(fp.read_bytes()).decode('ascii')
    except (OSError, ValueError):
        return ''


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description='V6: hien thi nham lan nhan dien tren Validation V5')
    ap.add_argument('--csv', type=Path, default=here / 'DU_DOAN_CHI_TIET_VALIDATION_V5.csv')
    ap.add_argument('--lock', type=Path, default=Path(r'D:\DoanHMvUD\external_field_pool_v2\EXTERNAL_SPLIT_LOCKED.csv'))
    ap.add_argument('--out', type=Path, default=here / 'KET_QUA_RASOAT_V6')
    args = ap.parse_args()
    rows = readcsv(args.csv)
    if not rows or not set(FIELDS).issubset(rows[0]):
        raise ValueError(f'CSV thieu cot bat buoc: {sorted(set(FIELDS) - set(rows[0] if rows else []))}')
    if len({x['path'].casefold() for x in rows}) != len(rows):
        raise ValueError('CSV V5 chua duong dan anh trung, can kiem tra!')
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.lock.is_file() and (here / 'EXTERNAL_SPLIT_LOCKED.csv').is_file():
        args.lock = here / 'EXTERNAL_SPLIT_LOCKED.csv'
    audit = read_lock(args.lock, rows)
    if audit['lock_found'] and not audit['ok']:
        raise ValueError('VI PHAM SPLIT VAL! ' + json.dumps(audit, ensure_ascii=False)[:800])

    labeled = [x for x in rows if x['true_disease'].strip()]
    wrong = [x for x in labeled if x['joint_ok'] != '1']
    suspect = []
    for x in wrong:
        prob = float(x['max_blend_probability'])
        problem = 'SAI_CAY' if x['crop_ok'] != '1' else 'DUNG_CAY_SAI_BENH'
        suspect.append({**x, 'nhom_loi': problem,
                        'uu_tien': 'CAO' if prob >= 0.8 else 'TRUNG_BINH',
                        'can_xac_minh_nhan': '', 'nhan_sau_khi_xem': '',
                        'ghi_chu_nguoi_xem': ''})
    suspect.sort(key=lambda x: (x['uu_tien'] != 'CAO', x['true_crop'] != 'Tomato', -float(x['max_blend_probability'])))
    writecsv(args.out / 'CAC_ANH_DU_DOAN_SAI_CAN_XEM.csv', suspect,
             FIELDS + ['nhom_loi', 'uu_tien', 'can_xac_minh_nhan', 'nhan_sau_khi_xem', 'ghi_chu_nguoi_xem'])

    confusion = Counter((x['true_crop'], x['true_disease'], x['pred_disease']) for x in wrong)
    grouped = [{'true_crop': t, 'true_disease': d, 'pred_disease': p, 'wrong_images': n}
               for (t, d, p), n in confusion.most_common()]
    writecsv(args.out / 'NHAM_LAN_CHI_TIET.csv', grouped,
             ['true_crop', 'true_disease', 'pred_disease', 'wrong_images'])

    perclass = []
    grouped_classes = defaultdict(list)
    for x in labeled:
        grouped_classes[x['true_disease']].append(x)
    for c, items in sorted(grouped_classes.items(), key=lambda kv: -len(kv[1])):
        perclass.append({'true_disease': c, 'total': len(items),
                         'joint_correct': sum(r['joint_ok'] == '1' for r in items),
                         'joint_accuracy_pct': accuracy(items, 'joint_ok'),
                         'crop_correct': sum(r['crop_ok'] == '1' for r in items),
                         'crop_accuracy_pct': accuracy(items, 'crop_ok')})
    writecsv(args.out / 'DO_CHINH_XAC_TUNG_BENH.csv', perclass,
             ['true_disease', 'total', 'joint_correct', 'joint_accuracy_pct',
              'crop_correct', 'crop_accuracy_pct'])
    summary = {'validation_images': len(rows), 'labeled_diseases': len(labeled),
               'unverified_diseases': len(rows) - len(labeled),
               'crop_correct': sum(x['crop_ok'] == '1' for x in rows),
               'crop_accuracy_pct': accuracy(rows, 'crop_ok'),
               'joint_correct': sum(x['joint_ok'] == '1' for x in labeled),
               'joint_accuracy_pct': accuracy(labeled, 'joint_ok'),
               'joint_wrong': len(wrong),
               'wrong_crop': sum(x['crop_ok'] != '1' for x in wrong),
               'right_crop_wrong_disease': sum(x['crop_ok'] == '1' for x in wrong),
               'wrong_joint_high_confidence_08': sum(float(x['max_blend_probability']) >= .8 for x in wrong),
               'model_disagreement_images': sum(x['model_disagreement'] == '1' for x in rows),
               'split_audit': audit,
               'warning': 'Accuracy is on reused Validation only, not new independent field images.'}
    (args.out / 'TONG_KET_RASOAT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),encoding='utf-8')

    htmlrows = []
    for i, x in enumerate(suspect, start=1):
        src = thumb_image(x['path'])
        imgtag = f'<img loading="lazy" src="{src}" alt="anh la" />' if src else '<div class="missing">Không có ảnh trong môi trường này</div>'
        htmlrows.append(f'''<div class="case">
<div class="preview">{imgtag}</div><div class="info"><strong>#{i} · {html.escape(x['true_crop'])}</strong>
<span class="tag">{html.escape(x['nhom_loi'])}</span>
<p><b>Nhãn hiện tại:</b> {html.escape(x['true_disease'])}<br>
<b>Dự đoán:</b> {html.escape(x['pred_disease'])}<br>
<b>Cây dự đoán:</b> {html.escape(x['pred_crop'])}<br>
<b>Điểm đầu ra:</b> {float(x['max_blend_probability']):.3f}<br>
<b>Hai mô hình cây bất đồng:</b> {'Có' if x['model_disagreement']=='1' else 'Không'}</p>
<small>{html.escape(x['path'])}</small></div></div>''')
    page = '''<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>V6 - Xem ảnh nhận diện sai</title>
<style>body{font:15px system-ui,Arial,sans-serif;background:#f7f8fb;color:#212638;margin:30px auto;max-width:1100px;padding:0 20px}
h1{font-size:26px}.note{padding:14px;background:#fff;border:1px solid #d8dde8;border-radius:8px;line-height:1.6}
.case{display:flex;gap:20px;padding:18px;margin:12px 0;border:1px solid #dce2ea;border-radius:12px;background:white}
.preview{flex:0 0 210px;height:175px}.preview img{width:100%;height:100%;object-fit:contain;background:#f5f6f8}
.missing{background:#f1f1f1;padding:20px}.info{min-width:0}.info p{line-height:1.6}.info small{overflow-wrap:anywhere;color:#677287}
.tag{margin-left:15px;background:#f3e8e8;color:#773b3b;border-radius:16px;padding:4px 9px;font-size:12px}
@media(max-width:650px){.case{flex-direction:column}.preview{flex:0 0 auto;width:100%;height:200px}}</style></head><body>
<h1>V6 · Rà soát ảnh nhận diện sai trên Validation</h1><div class="note">Đây là ảnh Validation của thử nghiệm V5, <b>không phải Test độc lập</b>.
Ảnh được nhúng trực tiếp khi chạy chương trình trên máy có dữ liệu gốc. Ưu tiên kiểm tra nhãn bằng kiến thức chuyên môn trước khi huấn luyện lại.
Sửa nhãn trong CSV rà soát, không chỉnh sửa trực tiếp nhãn gốc hay Test.</div>
''' + '\n'.join(htmlrows) + '</body></html>'
    (args.out / 'XEM_ANH_DU_DOAN_SAI.html').write_text(page, encoding='utf-8')
    print('=== V6 CHI PHAN TICH, KHONG HUAN LUYEN ===')
    print(f'Val: {len(rows)}, co nhan: {len(labeled)}, dung ca cay+benh: {summary["joint_correct"]}/{len(labeled)}')
    print(f'Sai: {len(wrong)}, dung cay sai benh: {summary["right_crop_wrong_disease"]}, sai cay: {summary["wrong_crop"]}')
    print('Sai nhung diem dau ra >=0.8:', summary['wrong_joint_high_confidence_08'])
    print('Split lock:', 'VAL OK' if audit.get('ok') else 'CHUA XAC MINH - can co lock')
    print('Bao cao o:', args.out)
    print('Mo file XEM_ANH_DU_DOAN_SAI.html bang trinh duyet.')


if __name__ == '__main__':
    main()
