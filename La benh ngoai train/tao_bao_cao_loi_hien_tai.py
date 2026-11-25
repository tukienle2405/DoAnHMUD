# -*- coding: utf-8 -*-
"""Analyze existing prediction CSV only; NEVER train or tune on Test."""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_csv(p):
    with open(p, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='BAO_CAO_LOI_97')
    args = ap.parse_args()
    data = read_csv(Path(args.csv))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labeled = [r for r in data if r['benh_theo_nhan_nguon'] not in ('', 'CHUA_XAC_MINH')]
    disease_groups = defaultdict(list)
    conf = Counter()
    for r in labeled:
        gt = r['benh_theo_nhan_nguon']
        pred = r['benh_97_khi_biet_dung_cay']
        disease_groups[gt].append(r)
        conf[(gt, pred)] += 1
    per = []
    for label, images in sorted(disease_groups.items()):
        true_disease = sum(int(r['dung_benh_khi_biet_cay']) for r in images)
        true_joint = sum(int(r['dung_ca_cay_va_benh']) for r in images)
        per.append({'disease': label, 'total': len(images),
                    'correct_known_crop': true_disease,
                    'known_crop_pct': round(100 * true_disease / len(images), 2),
                    'correct_joint': true_joint, 'joint_pct': round(100 * true_joint / len(images), 2)})
    write_csv(out / 'HIEU_QUA_TUNG_BENH.csv', per, ['disease', 'total', 'correct_known_crop', 'known_crop_pct', 'correct_joint', 'joint_pct'])
    by_pair = [{'true_disease': gt, 'pred_given_true_crop': pred, 'count': n}
               for (gt, pred), n in sorted(conf.items())]
    write_csv(out / 'NHAM_LAN_GIUA_CAC_BENH.csv', by_pair, ['true_disease', 'pred_given_true_crop', 'count'])
    suspicious = []
    for r in data:
        # Only detect overt cross-crop filename signals; not a reliable medical/plant diagnosis.
        filename = Path(r['path'].replace('\\', '/')).name.lower()
        true_crop = r['cay_that'].lower()
        other = None
        for crop in ('tomato', 'potato', 'pepper', 'corn'):
            if re.search(r'(?<![a-z])' + re.escape(crop) + r'(?![a-z])', filename) and crop != true_crop and not (crop == 'pepper' and true_crop == 'pepper__bell'):
                other = crop
                break
        if other is not None:
            suspicious.append({'path': r['path'], 'crop_in_label': r['cay_that'],
                               'other_crop_mentioned_in_filename': other,
                               'review_reason': 'Filename suggests another crop; human must check original image'})
    write_csv(out / 'ANH_CAN_KIEM_TRA_NHAN_THEO_TEN.csv', suspicious,
              ['path','crop_in_label', 'other_crop_mentioned_in_filename', 'review_reason'])
    tomato = [r for r in data if r['cay_that'] == 'Tomato']
    tomato_known = [r for r in labeled if r['cay_that'] == 'Tomato']
    tomato_two = sum(r['benh_97_khi_biet_dung_cay'] in ('Tomato_Early_blight','Tomato_Late_blight') for r in tomato)
    summary = {'total': len(data), 'labeled': len(labeled),
               'correct_joint': sum(int(r['dung_ca_cay_va_benh']) for r in labeled),
               'tomato_total': len(tomato), 'tomato_labeled': len(tomato_known),
               'tomato_predictions_as_early_or_late_when_given_true_crop': tomato_two,
               'tomato_correct_given_true_crop': sum(int(r['dung_benh_khi_biet_cay']) for r in tomato_known),
               'tomato_correct_joint': sum(int(r['dung_ca_cay_va_benh']) for r in tomato_known),
               'suspicious_filename_count': len(suspicious),
               'warning': 'Filename suspicions are for manual review only; they do not prove annotation errors.'}
    with open(out / 'TONG_QUAN_LOI.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
