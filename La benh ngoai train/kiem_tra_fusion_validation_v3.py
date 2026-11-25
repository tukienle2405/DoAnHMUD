# -*- coding: utf-8 -*-
"""V3: Kết hợp hai mô hình CHỈ trên Validation, tuyệt đối KHÔNG dùng Test để chọn trọng số.

Công thức: P(crop) = alpha * P18(crop) + (1-alpha) * sum_{disease in crop} P97(disease)
Bệnh được chọn bằng logit lớn nhất của model97 trong cây được dự đoán.

- Chỉ tìm alpha trên VAL, không sửa/đào tạo checkpoint.
- Duy trì old-validation crop accuracy trong ngưỡng cho phép.
- Tuyệt đối không sửa app Streamlit.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import nang_cap_benh_97_thuc_dia as v2

ALPHAS = [1.0, 0.90, 0.75, 0.50, 0.25, 0.0]


def get_old_crop_val(old_pool, count_per_crop, seed):
    """Đánh giá trên ảnh cũ có nhãn được duyệt, KHÔNG đoán nhãn nhóm Papaya."""
    grouped = defaultdict(list)
    for disease, items in old_pool['val'].items():
        crop = v2.crop_of(disease)
        for r in items:
            grouped[crop].append({'path': r['path'], 'label': crop, 'crop': crop, 'group': 'old_verified'})
    rng = random.Random(seed)
    chosen = []
    for crop in sorted(grouped):
        chosen.extend(rng.sample(grouped[crop], min(count_per_crop, len(grouped[crop]))))
    return chosen, sorted(set(v2.CROPS) - set(grouped))


def record_predictions(samples, m18, m97, names18, names97, device, batch_size, is_field=False):
    import torch
    from torch.utils.data import DataLoader
    from PIL import Image, ImageOps

    trans = v2.make_transform(train=False)
    class Dataset:
        def __len__(self): return len(samples)
        def __getitem__(self, i):
            with Image.open(samples[i]['path']) as im:
                x = trans(ImageOps.exif_transpose(im).convert('RGB'))
            return x

    loader = DataLoader(Dataset(), batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=(device.type == 'cuda'))
    m18.eval(); m97.eval()
    crop_to_indices = {crop: [i for i, c in enumerate(names97) if v2.crop_of(c) == crop] for crop in names18}
    if not all(crop_to_indices.values()):
        raise ValueError('Model97 khong co du cac nhan cho 18 cay')
    out = []
    cursor = 0
    with torch.inference_mode():
        for images in loader:
            images = images.to(device, non_blocking=True)
            l18 = m18(images)
            l97 = m97(images)
            p18 = torch.softmax(l18, dim=1)
            p97 = torch.softmax(l97, dim=1)
            p97_crop = torch.stack([p97[:, crop_to_indices[c]].sum(dim=1) for c in names18], dim=1)
            best_disease = torch.tensor(
                [[idxs[int(torch.argmax(l97[j, idxs]).item())] for idxs in crop_to_indices.values()]
                 for j in range(len(images))], device='cpu'
            ).tolist()
            for i in range(len(images)):
                r = samples[cursor + i]
                known = v2.FIELD_LABELS.get(r.get('group', '')) if is_field else None
                out.append({
                    'path': r['path'], 'crop_true': r['crop'], 'disease_true': known,
                    'p18': p18[i].cpu().tolist(),
                    'p97_crop': p97_crop[i].cpu().tolist(),
                    'best_disease_by_crop': [names97[j] for j in best_disease[i]],
                })
            cursor += len(images)
    return out


def eval_alpha(records, names18, alpha):
    crop_right = joint_right = n_known = 0
    detail = []
    for r in records:
        probs = [alpha * p + (1 - alpha) * q for p, q in zip(r['p18'], r['p97_crop'])]
        ix = max(range(len(probs)), key=lambda i: probs[i])
        crop_pred = names18[ix]
        disease_pred = r['best_disease_by_crop'][ix]
        crop_ok = int(crop_pred == r['crop_true'])
        crop_right += crop_ok
        disease_ok = None
        if r['disease_true'] is not None:
            disease_ok = int(disease_pred == r['disease_true'])
            joint_right += disease_ok
            n_known += 1
        detail.append({
            'path': r['path'], 'crop_true': r['crop_true'], 'crop_pred': crop_pred,
            'crop_correct': crop_ok,
            'disease_true': r['disease_true'] or 'CHUA_XAC_MINH',
            'disease_pred': disease_pred,
            'joint_correct': '' if disease_ok is None else disease_ok,
        })
    return {
        'crop_correct': crop_right, 'crop_total': len(records), 'crop_pct': v2.pct(crop_right, len(records)),
        'joint_correct': joint_right, 'joint_total': n_known, 'joint_pct': v2.pct(joint_right, n_known),
    }, detail


def main():
    p = argparse.ArgumentParser(description='Kiem tra crop probability fusion tren validation CHI')
    p.add_argument('--root', default=None)
    p.add_argument('--field_csv', default=None)
    p.add_argument('--manifest', default=None)
    p.add_argument('--model18', default=None)
    p.add_argument('--model97', default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--old_val_per_crop', type=int, default=16)
    p.add_argument('--max_old_crop_drop_pp', type=float, default=5.0)
    p.add_argument('--min_joint_gain_pp', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = p.parse_args()

    root = Path(args.root) if args.root else v2.find_root()
    out = Path(args.out) if args.out else root / 'training_runs' / 'field97_v3_fusion_val_only'
    out.mkdir(parents=True, exist_ok=True)
    field_csv = v2.find_field_csv(root, args.field_csv)
    manifest = v2.find_manifest(root, args.manifest)
    # 'test' appears only in the split integrity check. We NEVER read test images or run test inference.
    field_splits = v2.split_locked(v2.csv_rows(field_csv))
    field_val = [dict(r) for r in field_splits['val']]
    for r in field_val:
        r['label'] = r['crop']
    old_pool, _ = v2.scan_old_manifest(manifest)
    old_val, missing_old_crops = get_old_crop_val(old_pool, args.old_val_per_crop, args.seed)
    if not old_val:
        raise ValueError('Khong co old validation da xac minh')
    print('Field validation:', len(field_val), 'Old validation:', len(old_val))
    print('Khong co old val tin cay:', missing_old_crops)
    v2.check_existing_files(field_val, 'field validation')
    v2.check_existing_files(old_val, 'old validation')

    import torch
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    ckpt18 = Path(args.model18) if args.model18 else root / 'training_runs' / 'crop18_field_pilot_20261125_123457' / 'best_field_candidate.pth'
    ckpt97 = Path(args.model97) if args.model97 else root / 'training_runs' / 'field97_v2_safe' / 'field97_best_candidate.pth'
    print('Device:', device, 'Model18:', ckpt18, 'Model97 V2:', ckpt97, sep='\n')
    model18, names18 = v2.load_checkpoint(ckpt18, 18, device)
    model97, names97 = v2.load_checkpoint(ckpt97, 97, device)
    if not set(v2.FIELD_LABELS.values()).issubset(names97):
        raise ValueError('Khong khop nhan benh model97')
    field_preds = record_predictions(field_val, model18, model97, names18, names97, device, args.batch_size, True)
    old_preds = record_predictions(old_val, model18, model97, names18, names97, device, args.batch_size)
    results = []
    details = {}
    for alpha in ALPHAS:
        fm, fd = eval_alpha(field_preds, names18, alpha)
        om, _ = eval_alpha(old_preds, names18, alpha)
        results.append({'alpha': alpha, 'field_crop_pct': fm['crop_pct'],
                        'field_crop_correct': fm['crop_correct'],
                        'field_joint_pct': fm['joint_pct'], 'field_joint_correct': fm['joint_correct'],
                        'old_crop_pct': om['crop_pct'], 'old_crop_correct': om['crop_correct']})
        details[alpha] = fd
    baseline = next(x for x in results if x['alpha'] == 1.0)
    eligible = [x for x in results
                if x['field_joint_pct'] >= baseline['field_joint_pct'] + args.min_joint_gain_pp
                and x['field_crop_correct'] >= baseline['field_crop_correct']
                and x['old_crop_pct'] >= baseline['old_crop_pct'] - args.max_old_crop_drop_pp]
    chosen = max(eligible, key=lambda r: (r['field_joint_correct'], r['field_crop_correct'],
                                           r['old_crop_correct'], r['alpha'])) if eligible else None
    report = {
        'mode': 'VALIDATION_ONLY_NO_TEST_INFERENCE',
        'baseline_alpha1': baseline,
        'all_candidates': results,
        'chosen': chosen,
        'approved_for_further_independent_verification': chosen is not None,
        'missing_old_validation_crop': missing_old_crops,
        'warning': 'Old validation does not cover missing crops. VAL selection is not final real-world certification.',
        'model18': str(ckpt18), 'model97': str(ckpt97),
        'field_val_count': len(field_preds), 'old_val_count': len(old_preds),
        'no_test_used': True, 'no_auto_deploy': True,
    }
    v2.write_csv(out / 'SO_SANH_FUSION_VALIDATION.csv', results)
    v2.write_csv(out / 'DU_DOAN_VALIDATION_CHI_TIET.csv', details[chosen['alpha'] if chosen else 1.0])
    v2.save_json(out / 'TONG_KET_FUSION_VALIDATION.json', report)
    if chosen:
        v2.save_json(out / 'CONFIG_FUSION_CAN_KIEM_CHUNG.json', {
            'alpha': chosen['alpha'],
            'model18': str(ckpt18), 'model97': str(ckpt97),
            'not_tested_on_independent_new_holdout': True,
            'no_auto_deploy': True,
        })
    print('===== FUSION V3 VALIDATION ONLY =====')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print('Da luu:', out)

if __name__ == '__main__':
    main()
