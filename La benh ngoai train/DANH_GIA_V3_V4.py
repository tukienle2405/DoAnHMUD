# -*- coding: utf-8 -*-
"""V4: Read-only comparison of V2 baseline vs V3 crop on locked test or independent holdout.
No training, tuning, checkpoint writes, or app modification.
Existing locked Test has already been used during V2 development; its result is secondary,
NOT a substitute for an independent unseen holdout of all crops and diseases.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import nang_cap_benh_97_thuc_dia as v2

HERE = Path(__file__).resolve().parent

def normalize_path(path):
    return str(path).replace('/', '\\').strip().casefold()

def load_holdout(csv_path, locked, old_manifest, names18, names97):
    if not csv_path.exists():
        raise FileNotFoundError(f'Khong tim thay holdout CSV: {csv_path}')
    raw = v2.csv_rows(csv_path)
    needed = {'path', 'crop', 'disease', 'verified', 'source'}
    if not raw or not needed.issubset(raw[0]):
        raise ValueError('CSV can cac cot: path,crop,disease,verified,source')
    known = set(normalize_path(r['path']) for r in locked)
    # Check original-manifest paths too; checking exact file bytes against the old corpus
    # is deliberately NOT claimed because that would require a separate image-hash audit.
    with old_manifest.open(newline='', encoding='utf-8-sig') as f:
        known.update(normalize_path(r['path']) for r in csv.DictReader(f))
    good, hashseen, missing = [], {}, []
    for i, r in enumerate(raw, start=2):
        if not (r.get('path') or '').strip():
            continue  # the 97-class checklist has intentionally blank rows
        if r['verified'].strip().upper() not in {'YES', 'Y', 'TRUE', '1'}:
            continue  # only human-verified rows can enter published metrics
        image_path = Path(r['path'].strip())
        crop = r['crop'].strip()
        disease = r['disease'].strip()
        if crop not in names18:
            raise ValueError(f'Dong {i}: crop khong thuoc 18 cay: {crop}')
        if disease and (disease not in names97 or v2.crop_of(disease) != crop):
            raise ValueError(f'Dong {i}: disease khong phu hop crop: {disease}')
        if not r['source'].strip():
            raise ValueError(f'Dong {i}: source can ghi ro nguon anh')
        if normalize_path(image_path) in known:
            raise ValueError(f'Ro ri duong dan anh Train/Val/Test hoac du lieu cu: {image_path}')
        if not image_path.is_file():
            missing.append(str(image_path))
            continue
        hashval = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if hashval in hashseen:
            raise ValueError(f'Trung file trong holdout: {image_path} va {hashseen[hashval]}')
        hashseen[hashval] = str(image_path)
        good.append({'path':str(image_path), 'crop':crop, 'label':crop,
                     'disease':disease, 'source':r['source'].strip(), 'hash':hashval})
    if missing:
        raise FileNotFoundError(f'{len(missing)} anh verified khong tim thay, vi du: {missing[0]}')
    if not good:
        raise ValueError('Khong co anh nao co verified=YES va path hop le. Can dien mau CSV.')
    return good

def eval_joint(models, names18, names97, rows, device, batch_size):
    import torch
    crop_old, crop_new, disease_model = models
    mapper_crop = {s:i for i,s in enumerate(names18)}
    labeled = [r for r in rows if r.get('disease')]
    ds = v2.LeafDataset(rows, mapper_crop, False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                     num_workers=0, pin_memory=(device.type == 'cuda'))
    crop_old.eval(); crop_new.eval(); disease_model.eval()
    recs, pos = [], 0
    per_crop = defaultdict(Counter)
    with torch.inference_mode():
        for batch_x, _, _ in dl:
            x = batch_x.to(device, non_blocking=True)
            pold = crop_old(x).argmax(dim=1).cpu().tolist()
            pnew = crop_new(x).argmax(dim=1).cpu().tolist()
            logits97 = disease_model(x)
            for j in range(len(pold)):
                src = rows[pos]; pos += 1
                true_crop = src['crop']
                old_crop, new_crop = names18[pold[j]], names18[pnew[j]]
                out_old = names97[v2.crop_logits_predict(logits97[j], names97, old_crop)]
                out_new = names97[v2.crop_logits_predict(logits97[j], names97, new_crop)]
                disease = src.get('disease', '')
                group = per_crop[true_crop]
                group['crop_total'] += 1
                group['crop_old'] += int(old_crop == true_crop)
                group['crop_new'] += int(new_crop == true_crop)
                if disease:
                    group['disease_total'] += 1
                    group['joint_old'] += int(out_old == disease)
                    group['joint_new'] += int(out_new == disease)
                    group['oracle_disease'] += int(names97[v2.crop_logits_predict(logits97[j], names97, true_crop)] == disease)
                recs.append({'path':src['path'], 'true_crop':true_crop, 'true_disease':disease,
                             'old_crop':old_crop, 'new_crop':new_crop,
                             'old_disease':out_old, 'new_disease':out_new,
                             'old_crop_ok':int(old_crop==true_crop),
                             'new_crop_ok':int(new_crop==true_crop),
                             'old_joint_ok':int(bool(disease) and out_old==disease),
                             'new_joint_ok':int(bool(disease) and out_new==disease),
                             'source':src.get('source','')})
    def summarize(column, filter_labeled=False):
        subset = [r for r in recs if r['true_disease']] if filter_labeled else recs
        n=len(subset)
        k=sum(r[column] for r in subset)
        return {'correct':k,'total':n,'pct':v2.pct(k,n)}
    metrics = {
        'crop_old_all': summarize('old_crop_ok'),
        'crop_new_all': summarize('new_crop_ok'),
        'joint_old_labeled': summarize('old_joint_ok',True),
        'joint_new_labeled': summarize('new_joint_ok',True),
        'delta_joint_pp': round((sum(r['new_joint_ok']-r['old_joint_ok'] for r in recs if r['true_disease'])*100/len(labeled)),2) if labeled else None,
        'disease_labeled':len(labeled), 'crop_coverage':len(per_crop),
        'total_crops_required':18,'total_diseases_required':97,
        'disease_coverage':len({r['true_disease'] for r in recs if r['true_disease']}),
        'missing_crops':sorted(set(names18)-set(per_crop)),
        'missing_diseases':sorted(set(names97)-{r['true_disease'] for r in recs if r['true_disease']}),
        'per_crop':{c:dict(d) for c,d in sorted(per_crop.items())},
    }
    return metrics, recs

def main():
    ap=argparse.ArgumentParser(description='V4: check V3 crop + V2 disease; no training, no deploy')
    ap.add_argument('--mode', choices=['locked-test','holdout'], default='locked-test')
    ap.add_argument('--holdout_csv', default=None)
    ap.add_argument('--root', default=None)
    ap.add_argument('--field_csv', default=None)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--model18_old', default=None)
    ap.add_argument('--model18_v3', default=None)
    ap.add_argument('--model97_v2', default=None)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    a=ap.parse_args()
    if a.batch_size < 1: raise ValueError('batch_size phai >= 1')
    root=Path(a.root) if a.root else v2.find_root()
    decisions=root/'training_runs'/'TOM_TAT_V3_VALIDATION.json'
    if not decisions.is_file():
        raise FileNotFoundError(f'Khong thay tong ket V3 Validation: {decisions}')
    approved=json.loads(decisions.read_text(encoding='utf-8'))
    if approved.get('crop_training_validated') is not True:
        raise ValueError('V3 crop chua duoc Validation chap thuan. Khong danh gia tiep.')
    model18old=Path(a.model18_old) if a.model18_old else root/'training_runs'/'crop18_field_pilot_20261125_123457'/'best_field_candidate.pth'
    model18v3=Path(a.model18_v3) if a.model18_v3 else root/'training_runs'/'field18_v3_safe'/'field18_best_candidate_v3.pth'
    model97=Path(a.model97_v2) if a.model97_v2 else root/'training_runs'/'field97_v2_safe'/'field97_best_candidate.pth'
    import torch
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA khong kha dung')
    print('Device:',device,flush=True)
    print('Models:',model18old,model18v3,model97,sep='\n - ',flush=True)
    m18old, names18=v2.load_checkpoint(model18old,18,device)
    m18v3, names18new=v2.load_checkpoint(model18v3,18,device)
    m97,names97=v2.load_checkpoint(model97,97,device)
    if names18new != names18:
        raise ValueError('Khong khop thu tu class_names giua 2 model 18 cay')
    field_csv=v2.find_field_csv(root,a.field_csv)
    locked=v2.csv_rows(field_csv)
    splits=v2.split_locked(locked)
    if a.mode=='locked-test':
        rows=[{'path':r['path'],'crop':r['crop'],'label':r['crop'],
               'disease':v2.FIELD_LABELS.get(r['group'],''),
               'source':'PlantDoc_locked_test'} for r in splits['test']]
        v2.check_existing_files(rows,'Locked TEST')
        disclaimer='Da xem ket qua locked test o V2. KHONG duoc goi la holdout hoan toan moi.'
        tag='LOCKED_TEST_THAM_KHAO'
    else:
        csv_path=Path(a.holdout_csv) if a.holdout_csv else HERE/'MAU_HOLDOUT_97_LOP.csv'
        manifest=v2.find_manifest(root,a.manifest)
        rows=load_holdout(csv_path,locked,manifest,names18,names97)
        disclaimer='Chua kiem tra trung anh gan giong hoac trung voi corpus cu bang perceptual hash; can audit nguon anh thu cong.'
        tag='HOLDOUT_MOI'
    metrics, preds=eval_joint((m18old,m18v3,m97),names18,names97,rows,device,a.batch_size)
    out=root/'training_runs'/'field97_v4_external_verification'/tag
    out.mkdir(parents=True,exist_ok=True)
    all_metrics={'mode':a.mode,'models':{'old18':str(model18old),'new18':str(model18v3),'disease97':str(model97)},
                 'disclaimer':disclaimer,'metrics':metrics,
                 'streamlit_not_modified':True,'no_training':True,'no_auto_deploy':True,
                 'certified_18_crops_97_diseases':False}
    v2.save_json(out/'KET_QUA_V4.json',all_metrics)
    v2.write_csv(out/'DU_DOAN_CHI_TIET_V4.csv',preds)
    print('===== BAO CAO V4 =====',flush=True)
    print(json.dumps(all_metrics,indent=2,ensure_ascii=False),flush=True)
    print('Files:',out,flush=True)
    if a.mode=='holdout':
        print('Chua du 18 cay/97 benh: phai bo sung anh thuc dia co xac minh nhan, kiem tra gan trung.',flush=True)
    print('KHONG thay doi checkpoint hay Streamlit.',flush=True)

if __name__=='__main__':
    main()
