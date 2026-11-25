# -*- coding: utf-8 -*-
"""V3: Fine-tune 18 crops on locked field TRAIN, keep disease V2 fixed.

Selection by field VALIDATION crop and 18-crop+97-disease joint accuracy.
Never uses field TEST for model/parameter selection. Original checkpoints untouched.
Old replay from disease-verified original data; Papaya has no verified data and
therefore cannot be certified by this evaluation. No auto-deploy.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import nang_cap_benh_97_thuc_dia as v2
from kiem_tra_fusion_validation_v3 import get_old_crop_val


def old_train_by_crop(old_pool):
    by = defaultdict(list)
    for disease, items in old_pool['train'].items():
        c = v2.crop_of(disease)
        by[c].extend({'path': r['path'], 'label': c, 'crop': c, 'source_domain': 'old'} for r in items)
    return by


def sample_old_by_crop(by_crop, rng, n):
    crops = sorted(c for c, items in by_crop.items() if items)
    if not crops:
        raise ValueError('Khong co old train da xac minh')
    return [rng.choice(by_crop[rng.choice(crops)]) for _ in range(n)]


def stats(model18, model97, names18, names97, field_val_all, field_val_labeled,
          old_val, device, batch_size):
    all_crop = v2.evaluate_old(model18, names18, field_val_all, device, batch_size)
    old_crop = v2.evaluate_old(model18, names18, old_val, device, batch_size)
    joint, detail = v2.evaluate_field(model97, model18, names97, names18,
                                      field_val_labeled, device, batch_size, True)
    return all_crop, old_crop, joint, detail


def main():
    parser = argparse.ArgumentParser(description='Fine tune 18 crops: field TRAIN + verified original replay; VAL-only selection')
    parser.add_argument('--root', default=None)
    parser.add_argument('--field_csv', default=None)
    parser.add_argument('--manifest', default=None)
    parser.add_argument('--model18', default=None)
    parser.add_argument('--model97', default=None, help='V2 checkpoint 97 diseases')
    parser.add_argument('--out', default=None)
    parser.add_argument('--mode', choices=['audit', 'train'], default='train')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--head_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--field_per_epoch', type=int, default=600)
    parser.add_argument('--old_per_epoch', type=int, default=600)
    parser.add_argument('--old_val_per_crop', type=int, default=16)
    parser.add_argument('--lr_head', type=float, default=0.0002)
    parser.add_argument('--lr_backbone', type=float, default=0.00001)
    parser.add_argument('--kd_weight', type=float, default=0.25)
    parser.add_argument('--max_old_crop_drop_pp', type=float, default=5.0)
    parser.add_argument('--min_joint_gain_pp', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    a = parser.parse_args()
    if a.batch_size < 2 or a.epochs < 1 or a.field_per_epoch < 1 or a.old_per_epoch < 1:
        raise ValueError('batch_size>=2, epochs>=1, field_per_epoch>=1, old_per_epoch>=1')
    root = Path(a.root) if a.root else v2.find_root()
    out = Path(a.out) if a.out else root / 'training_runs' / 'field18_v3_safe'
    out.mkdir(parents=True, exist_ok=True)
    field_csv = v2.find_field_csv(root, a.field_csv)
    split = v2.split_locked(v2.csv_rows(field_csv))
    manifest = v2.find_manifest(root, a.manifest)
    old_pool, _ = v2.scan_old_manifest(manifest)
    v2.ensure_no_overlap(split, old_pool)
    field_train_all = [{**r, 'label': r['crop'], 'source_domain': 'field'} for r in split['train']]
    field_val_all = [{**r, 'label': r['crop'], 'source_domain': 'field'} for r in split['val']]
    disease_val = v2.make_field_samples(split['val'])
    old_val, missing_val = get_old_crop_val(old_pool, a.old_val_per_crop, a.seed)
    by_old = old_train_by_crop(old_pool)
    missing_train = sorted(set(v2.CROPS) - set(by_old))
    report = {
        'field_train': len(field_train_all), 'field_val_crop': len(field_val_all),
        'field_val_disease': len(disease_val), 'old_val': len(old_val),
        'missing_old_train_crops': missing_train, 'missing_old_val_crops': missing_val,
        'field_train_crops': dict(Counter(r['crop'] for r in field_train_all)),
        'mode': 'FIELD_TRAIN_AND_VALIDATION_ONLY_NO_TEST',
        'critical_limitation': 'Some crops lack verified old labels. Do not claim validation of all 18 crops.',
    }
    v2.save_json(out / 'KIEM_TRA_DU_LIEU_V3.json', report)
    print('===== AUDIT V3 =====')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if a.mode == 'audit':
        return
    v2.check_existing_files(field_train_all, 'Field TRAIN')
    v2.check_existing_files(field_val_all, 'Field VAL')
    v2.check_existing_files(old_val, 'Old crop VAL')
    import torch
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device)
    ckpt18 = Path(a.model18) if a.model18 else root / 'training_runs' / 'crop18_field_pilot_20261125_123457' / 'best_field_candidate.pth'
    ckpt97 = Path(a.model97) if a.model97 else root / 'training_runs' / 'field97_v2_safe' / 'field97_best_candidate.pth'
    print('Device:', device, '\nModel 18 original:', ckpt18, '\nModel97 V2 fixed:', ckpt97, flush=True)
    model18, names18 = v2.load_checkpoint(ckpt18, 18, device)
    model97, names97 = v2.load_checkpoint(ckpt97, 97, device)
    model97.eval()
    for p in model97.parameters(): p.requires_grad_(False)
    assert set(names18) == v2.CROPS
    teacher, teacher_names = v2.load_checkpoint(ckpt18, 18, device)
    assert teacher_names == names18
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    mapper18 = {c:i for i,c in enumerate(names18)}

    field_start, old_start, joint_start, _ = stats(
        model18, model97, names18, names97, field_val_all, disease_val, old_val, device, a.batch_size)
    base = {'field_crop': field_start, 'field_joint': joint_start, 'old_crop': old_start}
    print('===== BASELINE VALIDATION =====')
    print(json.dumps(base, ensure_ascii=False, indent=2), flush=True)
    v2.save_json(out / 'BASELINE_V3_VALIDATION.json', base)

    best_score = (-1,-1,-1)
    best_row = None
    history=[]
    stale=0
    current_stage=None
    for epoch in range(1, a.epochs + 1):
        unfreeze = epoch > a.head_epochs
        if current_stage != unfreeze:
            optimizer = v2.setup_training(model18, unfreeze, a.lr_head, a.lr_backbone)
            current_stage = unfreeze
        rng=random.Random(a.seed+epoch)
        train_samples=v2.choose_field(field_train_all, rng, a.field_per_epoch) + sample_old_by_crop(by_old, rng, a.old_per_epoch)
        rng.shuffle(train_samples)
        v2.check_existing_files(train_samples, 'Epoch train')
        loader=v2.build_loader(train_samples, mapper18, device, a.batch_size, True)
        step=v2.train_epoch(model18, teacher, loader, optimizer, device,
                            device.type=='cuda', a.kd_weight)
        field_cur, old_cur, joint_cur, _=stats(model18, model97, names18, names97,
                                               field_val_all, disease_val, old_val,
                                               device, a.batch_size)
        old_ok=old_cur['pct']>=old_start['pct']-a.max_old_crop_drop_pp
        field_ok=field_cur['correct']>=field_start['correct']
        row={
            'epoch':epoch,'stage':'head' if not unfreeze else 'backbone+head',
            **step,
            'field_val_crop_correct':field_cur['correct'], 'field_val_crop_pct':field_cur['pct'],
            'field_val_joint_correct':joint_cur['joint_correct'], 'field_val_joint_pct':joint_cur['joint_pct'],
            'old_val_crop_pct':old_cur['pct'], 'old_retention_ok':old_ok, 'field_crop_not_worse':field_ok,
        }
        history.append(row)
        v2.write_csv(out/'LICH_SU_NANG_CAP_CAY_V3.csv',history)
        print('Epoch',epoch,'=>',row,flush=True)
        score=(joint_cur['joint_correct'],field_cur['correct'],old_cur['correct'])
        if old_ok and field_ok and score > best_score:
            best_score=score
            best_row=row
            torch.save({
                'model_state_dict':model18.state_dict(), 'class_names':names18,
                'source_checkpoint':str(ckpt18),'fixed_disease_checkpoint':str(ckpt97),
                'epoch':epoch,'field_val_joint':joint_cur,'field_val_crop':field_cur,
                'old_val_crop':old_cur, 'note':'NO FIELD TEST USED, NOT AUTO DEPLOYED',
            }, out/'field18_best_candidate_v3.pth')
            stale=0
        else:
            stale+=1
        if epoch>a.head_epochs and stale>=a.patience:
            print('Early stopping (VALIDATION), epoch',epoch,flush=True)
            break

    gain=(best_row['field_val_joint_pct']-joint_start['joint_pct']) if best_row else None
    approved=(best_row is not None and gain>=a.min_joint_gain_pp
              and best_row['field_val_crop_correct']>=field_start['correct'])
    final={
        'baseline':base,'best':best_row, 'gain_joint_pp':None if gain is None else round(gain,2),
        'approved_for_new_independent_verification':bool(approved),
        'candidate_path':str(out/'field18_best_candidate_v3.pth') if best_row else None,
        'missing_old_val_crops':missing_val,
        'no_field_test_used':True,'no_auto_deploy':True,
        'warning': '17/18 old crop val coverage if Papaya is missing. Need separately verified Papaya images for safe deployment.',
    }
    v2.save_json(out/'QUYET_DINH_V3_VALIDATION.json',final)
    print('===== V3 VALIDATION DECISION =====')
    print(json.dumps(final,ensure_ascii=False,indent=2),flush=True)
    if approved:
        print('Candidate chi cho phep KIEM CHUNG DOC LAP, CHUA SU DUNG CHINH THUC.')
    else:
        print('KHONG DAT VALIDATION: GIU MODEL 18 CAY CU.')

if __name__ == '__main__':
    main()
