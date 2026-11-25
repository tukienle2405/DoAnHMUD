# -*- coding: utf-8 -*-
"""V5: Ket hop XAC SUAT 2 model 18 cay, kiem tra chi tren Validation.

- Model 97 benh V2 duoc GIU CO DINH.
- Chi phan tich Validation da khoa va replay Validation du lieu cu da loc nhan.
- KHONG load bat ky anh field Test nao, KHONG train, KHONG ghi de checkpoint, KHONG sua app.
- Validation da duoc dung de chon V2/V3, nen ket qua V5 chi la tham do.
  Can bo anh doc lap chua su dung truoc khi dua vao ung dung.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import nang_cap_benh_97_thuc_dia as v2
from kiem_tra_fusion_validation_v3 import get_old_crop_val

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # 1=V3; 0=field pilot old


def evaluate(records, crops, alpha, with_disease):
    all_crop = joint = labeled = 0
    groups = defaultdict(Counter)
    detail = []
    for r in records:
        blended = [(1.0 - alpha) * o + alpha * n for o, n in zip(r['old_probs'], r['v3_probs'])]
        chosen_i = max(range(len(crops)), key=lambda i: blended[i])
        pred_crop = crops[chosen_i]
        true_crop = r['crop']
        crop_ok = pred_crop == true_crop
        all_crop += int(crop_ok)
        g = groups[true_crop]
        g['crop_total'] += 1
        g['crop_correct'] += int(crop_ok)
        pred_disease = r['best_disease_per_crop'][chosen_i] if with_disease else ''
        gt_disease = r.get('disease', '')
        joint_ok = None
        if with_disease and gt_disease:
            joint_ok = pred_disease == gt_disease
            joint += int(joint_ok)
            labeled += 1
            g['joint_total'] += 1
            g['joint_correct'] += int(joint_ok)
        detail.append({
            'path': r['path'], 'true_crop': true_crop, 'true_disease': gt_disease,
            'pred_crop': pred_crop, 'pred_disease': pred_disease,
            'crop_ok': int(crop_ok), 'joint_ok': '' if joint_ok is None else int(joint_ok),
            'max_blend_probability': round(max(blended), 6),
            'model_disagreement': int(r['old_top1'] != r['v3_top1']),
            'old_crop_pred': r['old_top1'], 'v3_crop_pred': r['v3_top1'],
        })
    return {
        'crop_correct': all_crop, 'crop_total': len(records), 'crop_pct': v2.pct(all_crop, len(records)),
        'joint_correct': joint if with_disease else None,
        'joint_total': labeled if with_disease else None,
        'joint_pct': v2.pct(joint, labeled) if with_disease else None,
        'per_crop': {k:dict(val) for k,val in sorted(groups.items())}
    }, detail


def predict_records(rows, old18, new18, m97, names18, names97, device, batch_size, with_disease):
    import torch
    from torch.utils.data import DataLoader
    crops_to_ids = {c: [i for i, disease in enumerate(names97) if v2.crop_of(disease)==c] for c in names18}
    if any(not inds for inds in crops_to_ids.values()):
        raise ValueError('Model97 thieu nhan benh cho mot hoac nhieu cay')
    ds = v2.LeafDataset(rows, {c:i for i,c in enumerate(names18)}, False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                    pin_memory=(device.type=='cuda'))
    out = []
    old18.eval(); new18.eval(); m97.eval()
    cursor = 0
    with torch.inference_mode():
        for x, _, _ in dl:
            x = x.to(device, non_blocking=True)
            po = torch.softmax(old18(x), dim=1).cpu().tolist()
            pn = torch.softmax(new18(x), dim=1).cpu().tolist()
            if with_disease:
                logits = m97(x)
                best_ids = []
                for c in names18:
                    ids = crops_to_ids[c]
                    best_ids.append([ids[int(i)] for i in logits[:, ids].argmax(dim=1).cpu().tolist()])
            for i in range(len(po)):
                row = rows[cursor+i]
                old_idx = max(range(len(names18)), key=lambda k: po[i][k])
                new_idx = max(range(len(names18)), key=lambda k: pn[i][k])
                out.append({
                    'path': row['path'], 'crop':row['crop'],
                    'disease': v2.FIELD_LABELS.get(row.get('group', ''), '') if with_disease else '',
                    'old_probs': po[i], 'v3_probs': pn[i],
                    'old_top1': names18[old_idx], 'v3_top1': names18[new_idx],
                    'best_disease_per_crop': []
                })
                if with_disease:
                    out[-1]['best_disease_per_crop'] = [names97[best_ids[j][i]] for j in range(len(names18))]
            cursor += len(po)
    if cursor != len(rows):
        raise RuntimeError('So luong du doan khong khop du lieu Validation')
    return out


def main():
    p = argparse.ArgumentParser(description='V5: thu ensemble old18 + V3 crop tren VALIDATION, khong TEST')
    p.add_argument('--root', default=None)
    p.add_argument('--field_csv', default=None)
    p.add_argument('--manifest', default=None)
    p.add_argument('--old18', default=None)
    p.add_argument('--v3_18', default=None)
    p.add_argument('--v2_97', default=None)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--old_val_per_crop', type=int, default=16)
    p.add_argument('--max_old_val_drop_pp', type=float, default=1.0)
    p.add_argument('--min_joint_gain_pp', type=float, default=1.0)
    p.add_argument('--max_group_loss', type=int, default=1,
                   help='So anh joint dung toi da duoc phep giam tren 1 nhom >= 10 anh VALIDATION')
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    a = p.parse_args()
    if a.batch_size < 1 or a.max_group_loss < 0:
        raise ValueError('batch_size >= 1, max_group_loss >= 0')
    root = Path(a.root) if a.root else v2.find_root()
    out = root/'training_runs'/'field97_v5_ensemble_val_only'
    out.mkdir(parents=True, exist_ok=True)
    field_csv = v2.find_field_csv(root,a.field_csv)
    manifest = v2.find_manifest(root,a.manifest)
    raw = v2.csv_rows(field_csv)
    split = v2.split_locked(raw)  # chi kiem tra metadata split, KHONG doc anh Test
    old_pool,_ = v2.scan_old_manifest(manifest)
    v2.ensure_no_overlap(split,old_pool)
    field_val = [{**r,'label':r['crop'],'source_domain':'field'} for r in split['val']]
    old_val,missing_old = get_old_crop_val(old_pool,a.old_val_per_crop,2026)
    if not field_val or not old_val: raise ValueError('Thieu anh VALIDATION')
    v2.check_existing_files(field_val,'field validation')
    v2.check_existing_files(old_val,'old replay validation')
    import torch
    dev = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    if dev.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA khong kha dung')
    p_old = Path(a.old18) if a.old18 else root/'training_runs'/'crop18_field_pilot_20261125_123457'/'best_field_candidate.pth'
    p_new = Path(a.v3_18) if a.v3_18 else root/'training_runs'/'field18_v3_safe'/'field18_best_candidate_v3.pth'
    p_disease = Path(a.v2_97) if a.v2_97 else root/'training_runs'/'field97_v2_safe'/'field97_best_candidate.pth'
    print('THIET BI:',dev,flush=True)
    print('MODEL old18:',p_old,'\nMODEL new18:',p_new,'\nMODEL 97:',p_disease,flush=True)
    old18, names_old = v2.load_checkpoint(p_old,18,dev)
    new18, names_new = v2.load_checkpoint(p_new,18,dev)
    m97,names97 = v2.load_checkpoint(p_disease,97,dev)
    if set(names_old) != set(names_new): raise ValueError('18 cay khong khop')
    # Chuan hoa thu tu xac suat (can neu thu tu class_names khac nhau)
    if names_old != names_new:
        class Reorder(torch.nn.Module):
            def __init__(self,m,indices):
                super().__init__(); self.m=m; self.indices=indices
            def forward(self,x): return self.m(x)[:,self.indices]
        new18 = Reorder(new18, [names_new.index(c) for c in names_old]).to(dev)
    print('Field VAL:',len(field_val),'Old VAL:',len(old_val),
          'Old missing crops:',missing_old,flush=True)
    field_recs=predict_records(field_val,old18,new18,m97,names_old,names97,dev,a.batch_size,True)
    old_recs=predict_records(old_val,old18,new18,m97,names_old,names97,dev,a.batch_size,False)
    results=[]; details={}
    for alpha in ALPHAS:
        fm, fd = evaluate(field_recs,names_old,alpha,True)
        om, _ = evaluate(old_recs,names_old,alpha,False)
        results.append({'alpha_v3':alpha,
                        'field_val_crop_correct':fm['crop_correct'],
                        'field_val_crop_pct':fm['crop_pct'],
                        'field_val_joint_correct':fm['joint_correct'],
                        'field_val_joint_pct':fm['joint_pct'],
                        'old_val_crop_correct':om['crop_correct'],
                        'old_val_crop_pct':om['crop_pct'],
                        'field_per_crop':fm['per_crop']})
        details[alpha]=fd
    base=next(r for r in results if r['alpha_v3']==1.0)  # V3 duoc Validation chon truoc
    eligible=[]
    for x in results:
        groups_ok=all(x['field_per_crop'].get(c,{}).get('joint_correct',0) >= g.get('joint_correct',0) - a.max_group_loss
                      for c,g in base['field_per_crop'].items() if g.get('joint_total',0)>=10)
        is_good=(x['alpha_v3']!=1.0
                 and x['field_val_joint_pct'] >= base['field_val_joint_pct'] + a.min_joint_gain_pp
                 and x['field_val_crop_correct'] >= base['field_val_crop_correct']
                 and x['old_val_crop_pct'] >= base['old_val_crop_pct'] - a.max_old_val_drop_pp
                 and groups_ok)
        x['passes_validation_rules'] = bool(is_good)
        x['per_crop_guard_ok'] = bool(groups_ok)
        if is_good: eligible.append(x)
    winner=max(eligible,key=lambda x:(x['field_val_joint_correct'],x['field_val_crop_correct'],x['old_val_crop_correct'],x['alpha_v3'])) if eligible else None
    def public_row(r): return {k:v for k,v in r.items() if k!='field_per_crop'}
    report={
        'mode':'VALIDATION_ONLY_NO_TEST_INFERENCE',
        'warning':'VALIDATION DA DUOC DUNG CHON V2/V3; day chi la phep thu noi bo. KHONG tuyen bo do chinh xac tren anh moi.',
        'v3_baseline':public_row(base),
        'all_alphas':[public_row(r) for r in results],
        'chosen_ensemble':public_row(winner) if winner else None,
        'decision':'VALIDATION_CANDIDATE_ONLY_REQUIRE_INDEPENDENT_HOLDOUT' if winner else 'KEEP_V3_BASELINE',
        'field_val_images':len(field_val),
        'field_val_disease_labeled':sum(bool(v2.FIELD_LABELS.get(r.get('group',''))) for r in field_val),
        'old_val_images':len(old_val),
        'old_val_missing_crops':missing_old,
        'models':{'old18':str(p_old),'v3_18':str(p_new),'v2_97':str(p_disease)},
        'no_test_images_opened':True,'checkpoints_untouched':True,'streamlit_untouched':True,
    }
    v2.save_json(out/'TONG_KET_V5_VALIDATION.json',report)
    v2.write_csv(out/'SO_SANH_TRONG_SO_V5.csv',[public_row(r) for r in results])
    v2.write_csv(out/'DU_DOAN_CHI_TIET_VALIDATION_V5.csv',details[winner['alpha_v3'] if winner else 1.0])
    if winner:
        v2.save_json(out/'CAU_HINH_ENSEMBLE_CHI_DE_KIEM_CHUNG.json',{
            'alpha_v3':winner['alpha_v3'],'alpha_old18':1-winner['alpha_v3'],
            'method':'softmax probability weighted average, then disease97 masked by predicted crop',
            'validation_only':True,'no_auto_deploy':True,
        })
    print('\n===== KET QUA V5 VALIDATION =====',flush=True)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print('Da luu:',out,flush=True)
    print('KHONG danh gia Test, KHONG thay doi Streamlit, KHONG ghi de checkpoint.',flush=True)

if __name__=='__main__':
    main()
