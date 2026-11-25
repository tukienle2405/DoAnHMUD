# -*- coding: utf-8 -*-
"""V12 - Holdout doc lap: V2 vs V11, giu mo hinh cay V3.

Khong mo TEST/VAL cu. Khong huan luyen, khong ghi de checkpoint, khong sua Streamlit.
Moi anh can duoc gan nhan tu nguon doc lap TRUOC khi chay evaluate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMAGE_DIR = HERE / 'ANH_MOI_DOC_LAP'
OUTPUT_DIR = HERE / 'KET_QUA_V12'
LABEL_CSV = HERE / 'MAU_GAN_NHAN_V12.csv'
M18 = Path(r'D:\DoanHMvUD\training_runs\field18_v3_safe\field18_best_candidate_v3.pth')
M2 = Path(r'D:\DoanHMvUD\training_runs\field97_v2_safe\field97_best_candidate.pth')
M11 = ROOT / 'V11' / 'KET_QUA_V11' / 'V11_CANDIDATE_ONLY_NOT_DEPLOY.pth'
LOCKED = ROOT / 'V10' / 'EXTERNAL_SPLIT_LOCKED.csv'
MANIFEST = ROOT / 'V10' / 'dataset_manifest_SPLIT_DU_KIEN(1).csv'
EXTS = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
CROPS = {'Banana','Cashew','Chili','Coffee','Corn','Durian','Guava','Jackfruit',
         'Lemon','Mango','Papaya','Pepper__bell','Potato','Rice','Rubber','Sugarcane','Tea','Tomato'}
FIELDS = ['relative_path','true_class','label_verified','source','source_id','collection_date','notes']


def save_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def save_csv(p, rows, cols):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def read_csv(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def crop_of(s):
    if s.startswith('Pepper__bell___'):return 'Pepper__bell'
    if s.startswith('Tomato_'):return 'Tomato'
    return s.split('___',1)[0]


def all_images():
    IMAGE_DIR.mkdir(parents=True,exist_ok=True)
    return sorted((p for p in IMAGE_DIR.rglob('*') if p.is_file() and p.suffix.lower() in EXTS),
                  key=lambda p:p.relative_to(IMAGE_DIR).as_posix().casefold())


def relative_of(path):return path.relative_to(IMAGE_DIR).as_posix()


def safe_relative_path(relative):
    relative=relative.replace('\\','/')
    from pathlib import PurePosixPath
    p=PurePosixPath(relative)
    if not relative or p.is_absolute() or '..' in p.parts or ':' in relative:
        raise ValueError('Duong dan nguy hiem/ngoai ANH_MOI_DOC_LAP: '+relative)
    full=IMAGE_DIR.joinpath(*p.parts).resolve()
    if not full.is_relative_to(IMAGE_DIR.resolve()):
        raise ValueError('Anh nam ngoai ANH_MOI_DOC_LAP: '+relative)
    return full


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def make_label_list(names):
    save_csv(HERE/'DANH_SACH_97_NHAN_V12.csv',
             [{'true_class':n,'crop':crop_of(n)} for n in names],['true_class','crop'])


def load_names(path, count, required=True):
    if not path.is_file():
        if required:raise FileNotFoundError('Khong tim thay checkpoint: '+str(path))
        return None
    import torch
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(ckpt,dict) or 'model_state_dict' not in ckpt or 'class_names' not in ckpt:
        raise ValueError('Checkpoint sai dinh dang: '+str(path))
    names=list(ckpt['class_names'])
    if len(names)!=count or len(set(names))!=count:
        raise ValueError(f'Checkpoint {path} khong dung so nhan ({count})')
    if {crop_of(n) for n in names}!=CROPS:
        raise ValueError('Khong trung 18 cay: '+str(path))
    return names


def preflight(args, need_all=False):
    OUTPUT_DIR.mkdir(exist_ok=True)
    states={'mode':'CHECK_ONLY_NO_IMAGES','v2':str(args.v2), 'v11':str(args.v11),
            'v3_18':str(args.v3), 'images_dir':str(IMAGE_DIR),
            'v2_present':args.v2.is_file(),'v11_present':args.v11.is_file(),
            'v3_present':args.v3.is_file(), 'image_count':len(all_images()),
            'no_old_validation_or_test_opened':True,'no_training':True,
            'warning':'Chua co tap anh moi duoc xac minh. Khong co Accuracy cho V12.'}
    try:
        names11=load_names(args.v11,97,required=need_all)
        if names11:make_label_list(names11)
        if need_all:
            names2=load_names(args.v2,97)
            names18=load_names(args.v3,18)
            if set(names2)!=set(names11):raise ValueError('V2 va V11 khong cung tap nhan 97 lop')
            if set(names18)!=CROPS:raise ValueError('V3 khong du 18 cay')
            states['checkpoint_class_compatible']=True
        states['check_ok']=all([states['v2_present'],states['v11_present'],states['v3_present']])
    except Exception as e:
        states['check_ok']=False
        states['problem']=f'{type(e).__name__}: {e}'
        save_json(OUTPUT_DIR/'KIEM_TRA_V12.json',states)
        raise
    save_json(OUTPUT_DIR/'KIEM_TRA_V12.json',states)
    return states


def scan(args):
    states=preflight(args,need_all=False)
    old={}
    if LABEL_CSV.is_file():
        for row in read_csv(LABEL_CSV):
            if row.get('relative_path'):
                old[row['relative_path'].casefold()]=row
    images=all_images()
    rows=[]
    for img in images:
        rel=relative_of(img)
        r=dict.fromkeys(FIELDS,'')
        r.update({k:v for k,v in old.get(rel.casefold(),{}).items() if k in FIELDS})
        r['relative_path']=rel
        rows.append(r)
    # Preserve existing annotations even when files have been temporarily moved;
    # do not overwrite them silently. Evaluation will reject missing files.
    unseen=[v for k,v in old.items() if k not in {r['relative_path'].casefold() for r in rows}]
    if unseen:
        rows.extend(unseen)
        print(f'CANH BAO: {len(unseen)} dong CSV co anh khong con tren o dia. Khong xoa nhan.',flush=True)
    save_csv(LABEL_CSV,rows,FIELDS)
    report={'mode':'SCAN_NO_EVALUATION','images_found':len(images),
            'csv_rows':len(rows),'unavailable_previous_paths':len(unseen),
            'model_files_present':states['check_ok'], 'annotation_csv':str(LABEL_CSV),
            'labels_completed_and_verified':sum(r.get('label_verified','').strip().upper()=='YES' and bool(r.get('true_class','').strip()) for r in rows),
            'warning':'SCANNED only. Khong co du doan hay Accuracy. Anh va nhan moi can doc lap voi du lieu cu.'}
    save_json(OUTPUT_DIR/'TONG_KET_CHUAN_BI_V12.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    return report


def normcase_path(p):return str(p).replace('/','\\').strip().casefold()


def previous_paths():
    lookup=set();found=[]
    for p in [LOCKED, MANIFEST]:
        if not p.is_file():continue
        found.append(str(p))
        with p.open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                a=row.get('path','')
                if a:lookup.add(normcase_path(a))
    return lookup,found


def audit_new(rows):
    # Do not infer any label. Ground-truth must be entered by user from a separate source.
    images=all_images()
    on_disk={relative_of(p).casefold():p for p in images}
    refs,loaded=previous_paths()
    errors=[]; seen=set();seen_sha={};paths=[];near=[]
    for r in rows:
        rel=r.get('relative_path','').strip().replace('\\','/')
        if not rel:errors.append('Dong CSV thieu relative_path');continue
        if rel.casefold() in seen:errors.append('CSV co path trung: '+rel);continue
        seen.add(rel.casefold())
        try:path=safe_relative_path(rel)
        except Exception as e:errors.append(str(e));continue
        if rel.casefold() not in on_disk or not path.is_file():
            errors.append('Thieu file anh: '+rel);continue
        if normcase_path(path) in refs:errors.append('Path trung bo cu: '+rel)
        label=r.get('true_class','').strip()
        if not label:errors.append('Chua gan nhan: '+rel)
        if r.get('label_verified','').strip().upper()!='YES':
            errors.append('Nhan chua duoc xac minh YES: '+rel)
        if not r.get('source','').strip():errors.append('Chua ghi nguon: '+rel)
        if not r.get('source_id','').strip():errors.append('Chua ghi ma nguon/lan chup: '+rel)
        if not r.get('collection_date','').strip():errors.append('Chua ghi ngay thu thap: '+rel)
        try:
            digest=sha256(path)
            if digest in seen_sha:errors.append('Anh trung byte: '+rel+' va '+seen_sha[digest])
            seen_sha[digest]=rel
        except OSError as e:errors.append('Khong the doc: '+str(e))
        paths.append({'relative_path':rel,'path':path,'true_class':label,
                      'source':r.get('source',''),'source_id':r.get('source_id',''),
                      'collection_date':r.get('collection_date','')})
    unlisted=set(on_disk)-seen
    if unlisted:errors.append(f'{len(unlisted)} anh trong ANH_MOI_DOC_LAP chua co dong trong CSV; hay chay --mode scan')
    if not loaded:
        errors.append('Khong tim thay CSV khoa/manifest V10 de kiem tra path trung du lieu cu')
    return paths,errors,{'original_metadata_checked':loaded,'new_images':len(images),'new_unique_sha256':len(seen_sha),
                         'notes':'Chua doi chieu duoc noi dung anh trung/gan trung voi toan bo bo cu; path va SHA chi kiem tra gioi han.'}


def construct_model(path,n,device):
    import torch
    from model_architecture import get_mobilenet_v2_model
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    names=list(checkpoint['class_names'])
    model=get_mobilenet_v2_model(num_classes=n,pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model=model.to(device).eval()
    return model,names


def prepare_batch(image_paths):
    from PIL import Image,ImageOps
    from torchvision import transforms
    transform=transforms.Compose([
        transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    samples=[]
    for p in image_paths:
        with Image.open(p) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            samples.append(transform(im))
    import torch
    return torch.stack(samples)


def within_crop(logits,names,crop):
    ids=[i for i,s in enumerate(names) if crop_of(s)==crop]
    if not ids:raise ValueError('Checkpoint khong co nhan cho cay '+crop)
    ranked=sorted(ids,key=lambda i:float(logits[i].item()),reverse=True)
    return names[ranked[0]],' | '.join(names[i] for i in ranked[:3])


def wilson(k,n,z=1.96):
    if n<1:return [None,None]
    p=k/n; d=1+z*z/n
    middle=(p+z*z/(2*n))/d
    ext=(z*math.sqrt((p*(1-p)+z*z/(4*n))/n))/d
    return [round(100*(middle-ext),2),round(100*(middle+ext),2)]


def accuracy(rows,key):
    n=len(rows);k=sum(int(r[key]) for r in rows)
    return {'correct':k,'total':n,'pct':round(100*k/n,2) if n else None,
            'wilson_95_pct':wilson(k,n)}


def metrics_by_group(rows,key_name,name_column):
    groups=defaultdict(list)
    for r in rows:groups[r[name_column]].append(r)
    out=[]
    for name,samples in sorted(groups.items()):
        a=accuracy(samples,key_name)
        out.append({'name':name,'n':a['total'],'correct':a['correct'],
                    'accuracy_pct':a['pct']})
    return out


def main():
    p=argparse.ArgumentParser(description='V12: doc anh moi, so sanh V2 vs V11; khong huan luyen')
    p.add_argument('--mode',choices=['check','scan','evaluate'],default='check')
    p.add_argument('--batch_size',type=int,default=8)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--v2',type=Path,default=M2)
    p.add_argument('--v11',type=Path,default=M11)
    p.add_argument('--v3',type=Path,default=M18)
    args=p.parse_args()
    if args.mode=='check':
        report=preflight(args)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        if not report['check_ok']:
            print('THIEU CHECKPOINT. Kiem tra duong dan file trong bao cao.',file=sys.stderr)
            sys.exit(2)
        print('KIEM TRA DUONG DAN CHECKPOINT THANH CONG. Chua doc anh.',flush=True)
        return
    if args.mode=='scan':scan(args);return
    if args.batch_size<1:raise ValueError('batch_size phai >=1')
    scan(args)
    if not LABEL_CSV.is_file():raise FileNotFoundError('Thieu '+str(LABEL_CSV))
    labeled=read_csv(LABEL_CSV)
    if not labeled:
        print('CHUA CO ANH MOI. V12 KHONG THE TINH ACCURACY.');return
    paths,errors,audit=audit_new(labeled)
    if errors:
        save_json(OUTPUT_DIR/'LOI_DU_LIEU_V12.json',
                  {'total_errors':len(errors),'first_100_errors':errors[:100],'audit':audit})
        print('DUNG: nhan/nguon du lieu chua hop le. Xem KET_QUA_V12\\LOI_DU_LIEU_V12.json',flush=True)
        for err in errors[:20]:print(' - '+err)
        sys.exit(2)
    preflight(args,need_all=True)
    import torch
    torch.set_grad_enabled(False)
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('PyTorch chua co CUDA. Hay dung --device cpu')
    device=torch.device('cuda' if (args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available())) else 'cpu')
    print('Dang su dung:',device,flush=True)
    m18,n18=construct_model(args.v3,18,device)
    m2,n2=construct_model(args.v2,97,device)
    m11,n11=construct_model(args.v11,97,device)
    if set(n2)!=set(n11) or set(n18)!=CROPS:raise ValueError('Danh sach nhan cua checkpoint khong dong nhat')
    bad=[x for x in paths if x['true_class'] not in n2]
    if bad:
        raise ValueError(f'{len(bad)} nhan khong nam trong 97 nhan. VD: '+bad[0]['true_class'])
    predictions=[]
    with torch.inference_mode():
        for start in range(0,len(paths),args.batch_size):
            items=paths[start:start+args.batch_size]
            inp=prepare_batch([s['path'] for s in items]).to(device)
            y18=m18(inp).cpu();y2=m2(inp).cpu();y11=m11(inp).cpu()
            for j,s in enumerate(items):
                pred_crop=n18[int(y18[j].argmax().item())]
                true_label=s['true_class'];true_crop=crop_of(true_label)
                ans2,top2=within_crop(y2[j],n2,pred_crop)
                ans11,top11=within_crop(y11[j],n11,pred_crop)
                oracle2,_=within_crop(y2[j],n2,true_crop)
                oracle11,_=within_crop(y11[j],n11,true_crop)
                predictions.append({
                    'relative_path':s['relative_path'],'true_crop':true_crop,
                    'true_class':true_label,'source':s['source'],'source_id':s['source_id'],
                    'pred_crop_v3':pred_crop,'correct_crop':int(pred_crop==true_crop),
                    'pred_v2':ans2,'correct_v2':int(ans2==true_label),
                    'pred_v11':ans11,'correct_v11':int(ans11==true_label),
                    'pred_v2_if_true_crop':oracle2,'correct_v2_if_true_crop':int(oracle2==true_label),
                    'pred_v11_if_true_crop':oracle11,'correct_v11_if_true_crop':int(oracle11==true_label),
                    'top3_v2':top2,'top3_v11':top11,
                    'prediction_differs':int(ans2!=ans11)})
            print(f'{min(start+args.batch_size,len(paths))}/{len(paths)} anh',flush=True)
    assert len(predictions)==len(paths)
    save_csv(OUTPUT_DIR/'DU_DOAN_TUNG_ANH_V12.csv',predictions,list(predictions[0]))
    both_correct=sum(r['correct_v2'] and r['correct_v11'] for r in predictions)
    v2_only=sum(r['correct_v2'] and not r['correct_v11'] for r in predictions)
    v11_only=sum(r['correct_v11'] and not r['correct_v2'] for r in predictions)
    neither=sum(not r['correct_v11'] and not r['correct_v2'] for r in predictions)
    by_crop=[]
    for crop in sorted(set(r['true_crop'] for r in predictions)):
        sub=[r for r in predictions if r['true_crop']==crop]
        by_crop.append({'crop':crop,'samples':len(sub),
                        'crop_correct':sum(r['correct_crop'] for r in sub),
                        'v2_joint_correct':sum(r['correct_v2'] for r in sub),
                        'v11_joint_correct':sum(r['correct_v11'] for r in sub),
                        'v2_pct':accuracy(sub,'correct_v2')['pct'],
                        'v11_pct':accuracy(sub,'correct_v11')['pct']})
    save_csv(OUTPUT_DIR/'SO_SANH_THEO_CAY_V12.csv',by_crop,
             ['crop','samples','crop_correct','v2_joint_correct','v11_joint_correct','v2_pct','v11_pct'])
    by_class=[]
    for cls in sorted(set(r['true_class'] for r in predictions)):
        sub=[r for r in predictions if r['true_class']==cls]
        by_class.append({'class':cls,'samples':len(sub),
                         'v2_joint_correct':sum(r['correct_v2'] for r in sub),
                         'v11_joint_correct':sum(r['correct_v11'] for r in sub),
                         'v2_pct':accuracy(sub,'correct_v2')['pct'],
                         'v11_pct':accuracy(sub,'correct_v11')['pct']})
    save_csv(OUTPUT_DIR/'SO_SANH_THEO_BENH_V12.csv',by_class,
             ['class','samples','v2_joint_correct','v11_joint_correct','v2_pct','v11_pct'])
    met2=accuracy(predictions,'correct_v2');met11=accuracy(predictions,'correct_v11')
    report={
        'mode':'V12_NEW_SAMPLES_USER_ATTESTED_LABELS_NO_TRAINING',
        'total_images':len(predictions),'crops_tested':len(by_crop),
        'crop_names_tested':[x['crop'] for x in by_crop],
        'classes_tested':len(by_class),'classes_untested':97-len(by_class),
        'crop_v3':accuracy(predictions,'correct_crop'),
        'joint_v2':met2,'joint_v11':met11,
        'difference_v11_minus_v2_pp':round(met11['pct']-met2['pct'],2),
        'disease_given_true_crop_v2':accuracy(predictions,'correct_v2_if_true_crop'),
        'disease_given_true_crop_v11':accuracy(predictions,'correct_v11_if_true_crop'),
        'paired_comparison':{'both_correct':both_correct,'v2_only_correct':v2_only,
                             'v11_only_correct':v11_only,'both_wrong':neither},
        'input_audit':audit,
        'all_tested_labels_user_attested_verified':True,
        'old_validation_and_test_not_used_as_v12_images':True,
        'checkpoint_unchanged':True,'streamlit_unchanged':True,
        'approved_for_deployment':False,
        'warnings':[
          'Danh gia chi tin cay neu anh thuc su moi, khong trung nguon voi Train/Val/Test cu.',
          'Chuoi label_verified=YES la xac nhan cua nguoi dung, khong phai chung nhan chuyen gia.',
          'Chua kiem tra trung/giong nhau voi toan bo anh cu theo noi dung (chi metadata path).',
          'Khong suy ra ket qua cho 97 lop neu lop chua co anh trong bo moi.',
          'Cac so lieu nho phai xem them khoang tin cay va bat dong mau.'
        ]}
    save_json(OUTPUT_DIR/'TONG_KET_V12.json',report)
    print('===== V12 HOAN THANH (CHUA TU DONG DUYET TRIEN KHAI) =====')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
