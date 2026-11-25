# -*- coding: utf-8 -*-
"""V9: read-only data audit of old 97-class TRAIN and locked PlantDoc TRAIN.
Does NOT train, change labels, inspect any Validation/Test images, or touch model files.
"""
from __future__ import annotations
import argparse
import base64
import csv
import html
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

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
UNVERIFIED_GROUPS = {'Bell_pepper leaf spot','Tomato leaf yellow virus'}


def crop(label: str) -> str:
    if label.startswith('Pepper__bell___'): return 'Pepper__bell'
    if label.startswith('Tomato_'): return 'Tomato'
    return label.split('___', 1)[0]


def save_csv(p, rows, fields):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8-sig', newline='') as h:
        w = csv.DictWriter(h, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def select_samples(rows, n, seed=20261125):
    # Stable deterministic sample; does not use model prediction or validation labels.
    rng = random.Random(seed)
    s = sorted(rows, key=lambda x: x['path'])
    rng.shuffle(s)
    # Place potential naming conflicts first for manual source-label checks.
    s.sort(key=lambda x: (not bool(filename_flags(x['path'], x['class'])), x['path']))
    return s[:n]


def filename_flags(path, label):
    name = Path(path).name.casefold().replace('_','-')
    flags=[]
    if 'speck' in name and label == 'Tomato_Bacterial_spot':
        flags.append('TEN_FILE_SPECK_NHAN_SPOT_CAN_XAC_MINH')
    if any(x in name for x in ['tobrfv','brown-rugose']) and 'mosaic' in label.casefold():
        flags.append('TEN_FILE_TOBRFV_NHAN_MOSAIC_CAN_XAC_MINH')
    if 'early-blight' in name and 'late_blight' in label.casefold():
        flags.append('TEN_FILE_EARLY_NHAN_LATE_CAN_XAC_MINH')
    if 'late-blight' in name and 'early_blight' in label.casefold():
        flags.append('TEN_FILE_LATE_NHAN_EARLY_CAN_XAC_MINH')
    return flags


def image_preview(path, max_side=360):
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            w, h = im.size
            im = ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((max_side, max_side))
            buff = io.BytesIO(); im.save(buff, 'JPEG', quality=72, optimize=True)
            return 'data:image/jpeg;base64,'+base64.b64encode(buff.getvalue()).decode(), w, h, ''
    except ImportError:
        return '',None,None,'CHUA_CAI_PILLOW'
    except Exception as exc:
        return '',None,None, type(exc).__name__


def main():
    ap=argparse.ArgumentParser(description='V9 read-only TRAIN data audit; no training/test')
    here=Path(__file__).resolve().parent
    ap.add_argument('--manifest',default=str(here/'dataset_manifest_SPLIT_DU_KIEN(1).csv'))
    ap.add_argument('--lock',default=str(here/'EXTERNAL_SPLIT_LOCKED.csv'))
    ap.add_argument('--v8',default=str(here/'BANG_BAO_PHU_97_LOP_V8.csv'))
    ap.add_argument('--output',default=str(here/'KET_QUA_V9_TRAIN'))
    ap.add_argument('--samples-per-class',type=int,default=8)
    ap.add_argument('--skip-images',action='store_true',help='Only metadata; no image opening')
    args=ap.parse_args()
    p_manifest=Path(args.manifest);p_lock=Path(args.lock);out=Path(args.output)
    if not p_manifest.is_file(): ap.error('Khong tim thay manifest: '+str(p_manifest))
    if not p_lock.is_file(): ap.error('Khong tim thay locked split: '+str(p_lock))
    if not 1 <= args.samples_per_class <= 30: ap.error('--samples-per-class phai 1..30')
    out.mkdir(parents=True, exist_ok=True)
    print('V9: CHI RA SOAT TRAIN, KHONG HUAN LUYEN/DOI NHAN/DUA VAO TEST')

    old=defaultdict(lambda: defaultdict(Counter))
    flagged_by_class=defaultdict(list)
    tomato_and_potato_train=defaultdict(list)
    notes=[]; n_old=0; all_classes=set()
    with p_manifest.open('r',newline='',encoding='utf-8-sig') as h:
        reader=csv.DictReader(h)
        req={'path','class','split','needs_label_review'}
        if not req.issubset(set(reader.fieldnames or [])):
            ap.error('Manifest thieu cot: '+str(req-set(reader.fieldnames or [])))
        for r in reader:
            n_old+=1; split=(r.get('split') or '').strip(); lab=r['class']; all_classes.add(lab)
            review=(r.get('needs_label_review') or 'UNKNOWN').strip().upper()
            old[lab][split][review] += 1
            # Never open val/test files or select their paths for inspection.
            if split == 'train':
                if review!='NO': flagged_by_class[lab].append(r)
                if lab.startswith('Tomato_') or lab.startswith('Potato___'):
                    tomato_and_potato_train[lab].append(r)
    print('Manifest rows:',n_old,'| Classes:',len(all_classes))

    field_by_label=Counter();field_by_crop=Counter();field_train_total=0;field_unverified=Counter()
    field_train_rows_by_label=defaultdict(list); field_train_unverified_rows=defaultdict(list)
    locked_split=Counter();lock_clusters=defaultdict(set);lock_bad=0;lock_missing=0
    with p_lock.open('r',newline='',encoding='utf-8-sig') as h:
        reader=csv.DictReader(h)
        needed={'path','group','crop','split','reason','cluster'}
        if not needed.issubset(set(reader.fieldnames or [])):
            ap.error('Split lock thieu cot: '+str(needed-set(reader.fieldnames or [])))
        for r in reader:
            split=(r.get('split') or '').strip();locked_split[split]+=1
            cl=(r.get('cluster') or '').strip()
            if cl and split in {'train','val','test'}:lock_clusters[cl].add(split)
            if split!='train': continue  # HARD GATE: never open val/test images
            if r.get('reason','')!='OK':lock_bad+=1;continue
            field_train_total+=1
            lab=FIELD_LABELS.get(r.get('group',''))
            if lab:
                field_by_label[lab]+=1;field_by_crop[crop(lab)]+=1
                field_train_rows_by_label[lab].append(r)
            else:
                field_unverified[r.get('group','')]+=1
                field_train_unverified_rows[r.get('group','')].append(r)
    leaks=[c for c,s in lock_clusters.items() if len(s)>1]
    if leaks: raise ValueError(f'DUNG: {len(leaks)} clusters cross split in locked CSV, refuse audit')
    print('Lock split:',dict(locked_split),'| labeled TRAIN:',sum(field_by_label.values()))

    old_v8={}
    pv8=Path(args.v8)
    if pv8.is_file():
        with pv8.open('r',newline='',encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):old_v8[r.get('lop_benh','')]=r

    # Prioritize missing old-train unflagged classes and high-confusion tomato/potato classes.
    priority_names = ['Banana___healthy','Guava___Dot','Sugarcane___Sett_rot']
    priority_names += sorted(x for x in all_classes if x.startswith('Papaya___'))
    priority_names += ['Tomato_Bacterial_spot','Tomato_Septoria_leaf_spot','Tomato_Leaf_Mold',
        'Tomato_Late_blight','Tomato_Early_blight','Tomato_healthy','Tomato__Tomato_mosaic_virus',
        'Potato___Early_blight','Potato___Late_blight']
    order={x:i for i,x in enumerate(dict.fromkeys(priority_names))}
    summary_rows=[];all_risk=[]
    for lab in sorted(all_classes):
        a=old[lab]
        old_train_no=a['train']['NO'];old_train_yes=a['train']['YES']
        external=field_by_label[lab]
        missing= old_train_no == 0
        risk = ('THIEU_TRAIN_KHONG_GAN_CAN_RA_SOAT' if missing else
                'CAN_XEM_LAI_NHAN_TRAIN' if old_train_yes else 'KHONG_CO_CO_HIEU_TRONG_MANIFEST')
        summary_rows.append({'lop_benh':lab,'loai_cay':crop(lab),
            'train_cu_khong_gan_can_ra_soat':old_train_no,
            'train_cu_gan_can_ra_soat':old_train_yes,
            'train_thuc_dia_co_nhan_theo_nhom':external,
            'val_cu_chi_dem_khong_xem_anh':sum(a['val'].values()),
            'test_cu_chi_dem_khong_xem_anh':sum(a['test'].values()),
            'co_hieu':risk,
            'loi_validation_v7_chi_tham_khao':old_v8.get(lab,{}).get('so_loi_validation_v7','')})
        if missing:all_risk.append(lab)
    summary_rows.sort(key=lambda r:(r['train_cu_khong_gan_can_ra_soat']!=0,
          order.get(r['lop_benh'],999),r['lop_benh']))
    save_csv(out/'BAO_PHU_97_LOP_V9.csv',summary_rows,list(summary_rows[0]))

    samples=[]
    for lab in dict.fromkeys(priority_names):
        # Only TRAIN, choose already-flagged train rows for missing classes.
        pool=(flagged_by_class.get(lab,[]) if lab in all_risk else
              tomato_and_potato_train.get(lab,[]))
        for r in select_samples(pool,args.samples_per_class):
            fl=filename_flags(r['path'],lab)
            samples.append({'lop_benh_goc':lab,'loai_cay':crop(lab),
                'nguon':'MANIFEST_TRAIN_CU','path':r['path'],
                'needs_label_review_goc':r.get('needs_label_review',''),
                'co_hieu_ten_file':'; '.join(fl),
                'co_hieu': 'LOP_CHUA_CO_TRAIN_KHONG_CAN_RA_SOAT' if lab in all_risk else 'LOP_DE_NHAM_LAN',
                'anh_doc_duoc':'', 'kich_thuoc':'',
                'xac_minh_nhan_boi_chuyen_gia':'',
                'nhan_duoc_chuyen_gia_xac_minh':'',
                'ghi_chu_nguon_nhan':''})

    # Inspect suspicious field TRAIN source labels; validation/test images are untouched.
    n_field_name_flags=0
    for lab in dict.fromkeys(priority_names):
        candidates=field_train_rows_by_label.get(lab,[])
        scored=sorted(candidates, key=lambda r:(not bool(filename_flags(r['path'],lab)),r['path']))
        chosen=scored[:3]
        extra=[x for x in scored if filename_flags(x['path'],lab) and x not in chosen]
        chosen+=extra[:20]
        for r in chosen:
            fl=filename_flags(r['path'],lab)
            if fl:n_field_name_flags+=1
            samples.append({'lop_benh_goc':lab,'loai_cay':crop(lab),
                'nguon':'PLANTDOC_FIELD_TRAIN','path':r['path'],
                'needs_label_review_goc':'NHAN_THEO_NHOM_NGUON',
                'co_hieu_ten_file':'; '.join(fl),
                'co_hieu':'TEN_FILE_BAT_DONG_NHAN_LOAI_BENH' if fl else 'MAU_THUC_DIA_CAN_DOI_CHIEU',
                'anh_doc_duoc':'', 'kich_thuoc':'',
                'xac_minh_nhan_boi_chuyen_gia':'',
                'nhan_duoc_chuyen_gia_xac_minh':'',
                'ghi_chu_nguon_nhan':''})
    for grp, rows in sorted(field_train_unverified_rows.items()):
        for r in select_samples([dict(x, **{'class':grp}) for x in rows],2):
            samples.append({'lop_benh_goc':'CHUA_XAC_MINH:'+grp,'loai_cay':r.get('crop',''),
                'nguon':'PLANTDOC_FIELD_TRAIN_UNVERIFIED','path':r['path'],
                'needs_label_review_goc':'KHONG_DUNG_LAM_NHAN_BENH',
                'co_hieu_ten_file':'', 'co_hieu':'NHOM_BENH_CHUA_XAC_MINH',
                'anh_doc_duoc':'','kich_thuoc':'',
                'xac_minh_nhan_boi_chuyen_gia':'',
                'nhan_duoc_chuyen_gia_xac_minh':'',
                'ghi_chu_nguon_nhan':''})

    # Optional embedded thumbnails for train-only samples. Images are not edited.
    previews=[];readable=0;missing_images=0;lowres=0
    for r in samples:
        src=Path(r['path'])
        if not src.is_file():
            r['anh_doc_duoc']='KHONG_TIM_THAY';missing_images+=1;previews.append('')
            continue
        if args.skip_images:
            r['anh_doc_duoc']='CHUA_KIEM_TRA';previews.append('');continue
        b,w,h,issue=image_preview(src)
        if b:
            r['anh_doc_duoc']='CO';r['kich_thuoc']=f'{w}x{h}';readable+=1
            if w<256 or h<256:lowres+=1
        else:
            r['anh_doc_duoc']=issue
        previews.append(b)
    sample_fields=['lop_benh_goc','loai_cay','nguon','path','needs_label_review_goc','co_hieu_ten_file','co_hieu','anh_doc_duoc','kich_thuoc','xac_minh_nhan_boi_chuyen_gia','nhan_duoc_chuyen_gia_xac_minh','ghi_chu_nguon_nhan']
    save_csv(out/'MAU_CAN_XAC_MINH_TRAIN_V9.csv',samples,sample_fields)

    # Static self-contained HTML for optional manual source verification; not an auto diagnosis.
    styles="""<style>body{font:15px system-ui,sans-serif;max-width:1160px;margin:25px auto;color:#1e2733;background:#f7f9fb;padding:12px}article{display:flex;gap:16px;background:white;border:1px solid #d9e0ea;border-radius:10px;padding:16px;margin:12px 0;overflow-wrap:anywhere}img{width:260px;height:195px;object-fit:contain;background:#f5f5f5}.info{flex:1}small{color:#536075}code{font-size:12px}@media(max-width:600px){article{display:block}}</style>"""
    body=['<!doctype html><html lang="vi"><meta charset="utf-8"><title>V9 · Rà soát ảnh Train</title>',styles,
        '<h1>V9 – Ảnh TRAIN cần xác minh</h1><p>Chỉ hiển thị ảnh TRAIN. Mọi nhãn chỉ là nhãn nguồn, KHÔNG phải kết luận bệnh. Không mở ảnh Validation/Test và không sửa nhãn gốc.</p>']
    for i,(r,prev) in enumerate(zip(samples,previews),1):
        photo=f'<img loading="lazy" src="{prev}" alt="Anh train">' if prev else '<div>Không có hình xem trước</div>'
        body.append(f'<article>{photo}<div class="info"><b>#{i}: {html.escape(r["lop_benh_goc"])}</b>'
            f'<p>Cờ dữ liệu: {html.escape(r["co_hieu"])}<br> Cờ tên file: {html.escape(r["co_hieu_ten_file"] or "không")}<br>'
            f'Tình trạng: {html.escape(r["anh_doc_duoc"])} ({html.escape(r["kich_thuoc"])})</p>'
            f'<small>{html.escape(r["path"])}</small></div></article>')
    body.append('</html>');(out/'XEM_ANH_TRAIN_V9.html').write_text('\n'.join(body),encoding='utf-8')

    summary={
        'mode':'TRAIN_ONLY_DATA_AUDIT',
        'manifest_rows':n_old,
        'old_class_count':len(all_classes),
        'old_train_not_flagged_for_label_review':sum(x['train_cu_khong_gan_can_ra_soat'] for x in summary_rows),
        'old_train_flagged_for_label_review':sum(x['train_cu_gan_can_ra_soat'] for x in summary_rows),
        'classes_without_unflagged_old_train':all_risk,
        'field_locked_split_counts_METADATA_ONLY':dict(locked_split),
        'field_train_rows_ok':field_train_total,
        'field_train_disease_labeled_by_source_group':sum(field_by_label.values()),
        'field_train_unverified_groups':dict(field_unverified),
        'field_train_disease_classes':len(field_by_label),
        'samples_selected_from_train_only':len(samples),
        'field_train_name_conflict_sample_count':n_field_name_flags,
        'sample_images_read':readable,
        'sample_images_missing':missing_images,
        'sample_images_low_resolution':lowres,
        'cross_split_clusters_in_locked_csv':len(leaks),
        'no_train_or_test_files_modified':True,
        'no_validation_or_test_images_opened':True,
        'model_checkpoint_not_touched':True,
        'disclaimer':'needs_label_review=NO means NOT flagged, NOT professionally verified. No relabel performed. Train samples only.'
    }
    (out/'TONG_KET_V9.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'BAO_CAO_V9.txt').write_text(
        'V9 / READ ONLY / TRAIN AUDIT\n\n'+
        f'97-class rows: {n_old}; classes: {len(all_classes)}\n'+
        f'Old classes without unflagged TRAIN: {len(all_risk)}\n'+
        '\n'.join(' - '+x for x in all_risk)+'\n\n'+
        f'Field TRAIN source-labeled: {sum(field_by_label.values())}, unverified: {sum(field_unverified.values())}\n'+
        f'TRAIN sample images: {len(samples)}, opened {readable}, missing {missing_images}, low res {lowres}\n\n'+
        'Important: NO label changes, NO training, NO test/val image opened. Not all NO flags prove verified labels.\n'+
        'Next: expert/source verification for suspicious TRAIN labels, collect new independent holdout for all 18 crops / 97 classes.\n',encoding='utf-8')
    print('===== XONG V9 =====')
    print('Risk classes missing unflagged old TRAIN:',len(all_risk))
    print('TRAIN samples:',len(samples),'| readable:',readable,'| missing:',missing_images)
    print('Output:',out)
    print('GUI LAI: TONG_KET_V9.json, BAO_PHU_97_LOP_V9.csv, MAU_CAN_XAC_MINH_TRAIN_V9.csv')
    print('NEU CAN XEM ANH: XEM_ANH_TRAIN_V9.html')

if __name__=='__main__':main()
