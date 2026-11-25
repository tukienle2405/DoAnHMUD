# -*- coding: utf-8 -*-
"""V11: 97-class HEAD-ONLY field pilot with V10 quarantine.

- Updates ONLY final classifier rows corresponding to the 13 PlantDoc source-label groups.
- Preserves all other 84 classifier rows and full feature extractor exactly.
- Reads only TRAIN and previously-used VALIDATION, never opens TEST images.
- Does not change V2/V3 checkpoints, original CSVs, or Streamlit.
- Does NOT claim verified ground-truth or generalization to new field images.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import nang_cap_benh_97_thuc_dia as v2

DEFAULT_BASE = Path(r'D:\DoanHMvUD\La benh ngoai train')
MODEL_18 = Path(r'D:\DoanHMvUD\training_runs\field18_v3_safe\field18_best_candidate_v3.pth')
MODEL_97 = Path(r'D:\DoanHMvUD\training_runs\field97_v2_safe\field97_best_candidate.pth')


def read_rows(p):
    if not p.is_file():
        raise FileNotFoundError(f'Thieu file: {p}')
    return v2.csv_rows(p)


def source_is_test_folder(path):
    # This is the original PlantDoc source folder name, NOT the custom locked split.
    cleaned = path.replace('/', '\\').casefold()
    return '\\plantdoc_khong_trung_sha256\\test\\' in cleaned


def pick(paths, n, seed):
    rng = random.Random(seed)
    names = sorted(paths)
    if not names:
        raise ValueError('Khong co lop Old Train hop le')
    result = []
    for i in range(n):
        c = names[i % len(names)]
        result.append(rng.choice(paths[c]))
    rng.shuffle(result)
    return result


def build_data(base):
    r = base / 'V10'
    candidate = read_rows(r / 'KET_QUA_V10' / 'TRAIN_CANDIDATE_V10.csv')
    field = read_rows(r / 'KET_QUA_V10' / 'TRAIN_FIELD_SAFE_V10.csv')
    lock = read_rows(r / 'EXTERNAL_SPLIT_LOCKED.csv')
    manifest = r / 'dataset_manifest_SPLIT_DU_KIEN(1).csv'
    if not manifest.is_file():
        raise FileNotFoundError(f'Thieu manifest tai {manifest}')
    cand_old = [r for r in candidate if r['origin']=='OLD97_MANIFEST' and r['split']=='train']
    cand_field = [r for r in candidate if r['origin']=='PLANTDOC_FIELD_LOCKED' and r['split']=='train']
    if len(cand_old)!=100536 or len(cand_field)!=590 or len(field)!=590:
        raise ValueError('V10 khong trung du lieu da kiem tra (100536 Old + 590 Field). DUNG lai!')
    safe = {r['path'].casefold() for r in field}
    if safe != {r['path'].casefold() for r in cand_field}:
        raise ValueError('FIELD_SAFE va CANDIDATE khong dong nhat!')
    if len(safe)!=len(field):
        raise ValueError('FIELD_SAFE co trung path')
    if any(x['label'] not in v2.FIELD_LABELS.values() for x in field):
        raise ValueError('FIELD_SAFE co label khong thuoc nhom anh xa V2')
    for r0 in lock:
        if r0['split']=='train' and r0['reason']!='OK':
            raise ValueError('Co dong locked train reason khac OK')
    locked_train = {r0['path'].casefold() for r0 in lock if r0['split']=='train' and r0['reason']=='OK'}
    locked_val = {r0['path'].casefold() for r0 in lock if r0['split']=='val' and r0['reason']=='OK'}
    locked_test = {r0['path'].casefold() for r0 in lock if r0['split']=='test' and r0['reason']=='OK'}
    if not safe.issubset(locked_train) or safe&locked_val or safe&locked_test:
        raise ValueError('V10 FIELD_SAFE vi pham chia Train/Val/Test da khoa')
    if {x['path'].casefold() for x in cand_old}&(locked_val|locked_test):
        raise ValueError('Old train trung duong dan Field val/test')
    # Additional conservative measure: exclude source folder named test from pilot.
    field_usable = [x for x in field if not source_is_test_folder(x['path'])]
    source_test_count = len(field)-len(field_usable)
    field_train = [{'path': x['path'], 'label': x['label'], 'source_domain':'field'} for x in field_usable]
    field_val = v2.make_field_samples([x for x in lock if x['split']=='val' and x['reason']=='OK'])
    field_val_path = {x['path'].casefold() for x in field_val}
    if field_val_path & safe:
        raise ValueError('Train/val overlap!')
    old_by_class = defaultdict(list)
    for x in cand_old:
        old_by_class[x['label']].append({'path': x['path'], 'label': x['label'], 'source_domain':'old'})
    # Only labels not flagged for review enter old validation, just like V2.
    val_by_class = defaultdict(list)
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        for x in csv.DictReader(f):
            if x['split']=='val' and x.get('needs_label_review', 'NO').upper() != 'YES':
                val_by_class[x['class']].append({'path': x['path'], 'label': x['class'], 'source_domain':'old'})
    old_val = v2.balanced_val(val_by_class, 3, seed=2026)
    no_train = sorted(set(x['class'] for x in v2.csv_rows(r/'KET_QUA_V10'/'BAO_PHU_97_LOP_V10.csv') if x['train_class_missing']=='YES'))
    stats = {'old_train_count':len(cand_old), 'field_v10_safe':len(field),
             'field_excluded_original_source_test_folder':source_test_count,
             'field_pilot_train':len(field_train), 'old_validation_sample':len(old_val),
             'field_validation_sample':len(field_val), 'field_pilot_classes':len(set(x['label'] for x in field_train)),
             'missing_provisional_train_classes':no_train,
             'warning':'11 class labels in V10 quarantine, NOT cleared for supervised training.'}
    return old_by_class, field_train, old_val, field_val, stats


def freeze_all_except_final_head_rows(model, names, target_names):
    import torch
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.classifier[-1]
    if not isinstance(head, torch.nn.Linear) or head.out_features != len(names):
        raise TypeError('Checkpoint head architecture does not match expected MobileNetV2')
    head.weight.requires_grad_(True)
    head.bias.requires_grad_(True)
    mask = torch.zeros(len(names), dtype=head.weight.dtype, device=head.weight.device)
    for name in target_names:
        mask[names.index(name)] = 1
    head.weight.register_hook(lambda grad: grad*mask[:,None])
    head.bias.register_hook(lambda grad: grad*mask)
    return head, mask


def train_epoch(model, teacher, samples, names, device, batch, lr, kd_weight):
    import torch
    import torch.nn.functional as F
    # Features and batch norm are intentionally EVAL: the backbone remains unchanged.
    model.eval(); teacher.eval()
    mapper = {x:i for i,x in enumerate(names)}
    loader = v2.build_loader(samples, mapper, device, batch, train=True)
    head = model.classifier[-1]
    # weight_decay=0 is crucial: it ensures masked rows remain EXACTLY unchanged.
    optim = torch.optim.AdamW([head.weight,head.bias],lr=lr,weight_decay=0)
    avg=0.; total=0
    for images, target, is_old in loader:
        images=images.to(device); target=target.to(device); is_old=is_old.to(device).bool()
        optim.zero_grad(set_to_none=True)
        logits=model(images)
        loss=F.cross_entropy(logits,target)
        if kd_weight>0 and is_old.any():
            with torch.no_grad(): tlog=teacher(images[is_old])
            temperature=2.0
            loss+=kd_weight*F.kl_div(F.log_softmax(logits[is_old]/temperature,dim=1),
                                       F.softmax(tlog/temperature,dim=1),reduction='batchmean')*temperature**2
        loss.backward(); optim.step()
        total+=len(target); avg+=float(loss.item())*len(target)
    return round(avg/max(total,1),5)


def main():
    p=argparse.ArgumentParser(description='V11: safe head-only 97-class candidate, no test opened')
    p.add_argument('--base', type=Path, default=DEFAULT_BASE)
    p.add_argument('--mode', choices=['check','train'], default='check')
    p.add_argument('--model18',type=Path,default=MODEL_18)
    p.add_argument('--model97',type=Path,default=MODEL_97)
    p.add_argument('--epochs',type=int,default=2)
    p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--old_per_epoch',type=int,default=300)
    p.add_argument('--field_per_epoch',type=int,default=300)
    p.add_argument('--lr',type=float,default=0.0001)
    p.add_argument('--kd_weight',type=float,default=0.5)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=p.parse_args()
    random.seed(2026)
    out=Path(__file__).resolve().parent / 'KET_QUA_V11'
    out.mkdir(exist_ok=True)
    old_train,field_train,old_val,field_val,stats=build_data(args.base)
    v2.save_json(out/'KIEM_TRA_DAU_VAO_V11.json',stats)
    print('V11 preflight:',json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
    if len(field_train)<300 or len(set(x['label'] for x in field_train))!=13:
        raise ValueError('Field pilot training incomplete')
    if args.mode=='check':
        print('KIEM TRA CSV THANH CONG. Khong doc anh va khong huan luyen.')
        return
    if args.epochs<1 or args.epochs>5 or args.batch_size<2:
        raise ValueError('V11 pilot requires 1-5 epochs, batch_size >= 2')
    import torch
    device=torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available()) or args.device=='cuda' else 'cpu')
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('Khong co CUDA trong PyTorch')
    print('Training device',device,flush=True)
    model97,names97=v2.load_checkpoint(args.model97,97,device)
    model18,names18=v2.load_checkpoint(args.model18,18,device)
    if len(set(names97))!=97 or {x['label'] for x in field_train}.difference(names97):
        raise ValueError('Checkpoint and training classes mismatch')
    # Teacher is a frozen copy of the V2 starting checkpoint.
    teacher=copy.deepcopy(model97).eval()
    for par in teacher.parameters():par.requires_grad_(False)
    for par in model18.parameters():par.requires_grad_(False)
    target_names=sorted({x['label'] for x in field_train})
    head,mask=freeze_all_except_final_head_rows(model97,names97,target_names)
    base_weight=head.weight.detach().clone();base_bias=head.bias.detach().clone()
    keep=(mask==0)
    for items,loc in ((old_val,'old validation'),(field_val,'field validation')):
        v2.check_existing_files(items,loc)
    baseline_field,_=v2.evaluate_field(teacher,model18,names97,names18,field_val,device,args.batch_size)
    baseline_old=v2.evaluate_old(teacher,names97,old_val,device,args.batch_size)
    print('BASELINE V2+V3 field validation:',baseline_field,flush=True)
    print('BASELINE old validation:',baseline_old,flush=True)
    history=[]; best_joint=baseline_field['joint_correct']; best=None
    for epoch in range(1,args.epochs+1):
        old=pick(old_train,args.old_per_epoch,2026+epoch)
        rng=random.Random(9026+epoch)
        take_field=rng.choices(field_train,k=args.field_per_epoch)
        sample=old+take_field;rng.shuffle(sample)
        v2.check_existing_files(sample,f'Epoch {epoch}: train')
        loss=train_epoch(model97,teacher,sample,names97,device,args.batch_size,args.lr,args.kd_weight)
        # Strong exact-retention check for frozen rows.
        if not torch.equal(head.weight.detach()[keep],base_weight[keep]) or not torch.equal(head.bias.detach()[keep],base_bias[keep]):
            raise RuntimeError('Frozen output classes unexpectedly changed! Abort.')
        field_score,_=v2.evaluate_field(model97,model18,names97,names18,field_val,device,args.batch_size)
        old_score=v2.evaluate_old(model97,names97,old_val,device,args.batch_size)
        old_drop=round(baseline_old['pct']-old_score['pct'],2)
        row={'epoch':epoch,'train_loss':loss,'field_crop_pct':field_score['crop_pct'],
             'field_joint_pct':field_score['joint_pct'],
             'field_joint_correct':field_score['joint_correct'],
             'field_given_crop_pct':field_score['given_true_crop_pct'],
             'old_val_pct':old_score['pct'],'old_val_drop_pp':old_drop,
             'meets_provisional_rules': bool(field_score['joint_correct']>best_joint and old_drop<=1.0 and
                 field_score['given_true_crop_pct']>=baseline_field['given_true_crop_pct'])}
        history.append(row)
        v2.write_csv(out/'LICH_SU_V11.csv',history)
        print('Epoch',epoch,row,flush=True)
        if row['meets_provisional_rules']:
            best_joint=field_score['joint_correct']; best=row
            torch.save({'model_state_dict':model97.state_dict(),'class_names':names97,
                        'adaptation':'V11_HEAD_ONLY_V10_FILTERED_INTERNAL_VALIDATION',
                        'parent_checkpoint':str(args.model97), 'epoch':epoch,
                        'validation_only':True,'not_approved_for_deployment':True},
                       out/'V11_CANDIDATE_ONLY_NOT_DEPLOY.pth')
    summary={'mode':'V11_INTERNAL_VALIDATION_ONLY','start_model97':str(args.model97),
             'start_model18':str(args.model18),'training_images_per_epoch':args.old_per_epoch+args.field_per_epoch,
             'field_training_images_available':len(field_train),'excluded_original_source_test_folder':
             stats['field_excluded_original_source_test_folder'],
             'no_original_checkpoints_overwritten':True,'no_test_images_opened':True,
             'frozen_other_classifier_classes':97-len(target_names),
             'baseline_field_val':baseline_field,'baseline_old_val':baseline_old,
             'best_provisional_epoch':best, 'approved_for_deployment':False,
             'needs_fresh_independent_holdout':True,
             'warning':'Validation was reused in V2/V3/V5; any improvement is ONLY exploratory.'}
    v2.save_json(out/'TONG_KET_V11.json',summary)
    print('===== TONG KET V11 =====')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print('Do NOT deploy candidate or use TEST for tuning.',flush=True)

if __name__=='__main__':
    main()
