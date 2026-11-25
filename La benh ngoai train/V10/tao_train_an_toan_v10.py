# -*- coding: utf-8 -*-
"""V10 - Tách tập Train dựa trên bằng chứng nhãn/nguồn, không thay dữ liệu gốc.

Chỉ dùng metadata train để tạo danh sách ứng viên + cần xác minh.
Không suy đoán bệnh từ ảnh, không train model, không đụng Test/Validation images.
"""
import argparse
import csv
import json
import ntpath
import re
from collections import Counter, defaultdict
from pathlib import Path

GROUP_TO_LABEL = {
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


def conflict_flags(path, group):
    """Heuristic only: filename is not a verified diagnosis."""
    fn = ntpath.basename(path).casefold().replace('_', '-')
    flags = []
    if 'speck' in fn and group == 'Tomato leaf bacterial spot':
        flags.append('FILE_SPECK_VS_GROUP_SPOT')
    if re.search(r'late[ -]?blight', fn) and group == 'Potato leaf early blight':
        flags.append('FILE_LATE_VS_GROUP_EARLY')
    if re.search(r'early[ -]?blight', fn) and group == 'Potato leaf late blight':
        flags.append('FILE_EARLY_VS_GROUP_LATE')
    if re.search(r'late[ -]?blight', fn) and group == 'Tomato Early blight leaf':
        flags.append('FILE_LATE_VS_GROUP_EARLY')
    if re.search(r'early[ -]?blight', fn) and group == 'Tomato leaf late blight':
        flags.append('FILE_EARLY_VS_GROUP_LATE')
    if ('tobrfv' in fn or 'brown-rugose' in fn) and group == 'Tomato leaf mosaic virus':
        flags.append('FILE_TOBRFV_VS_GROUP_MOSAIC')
    return flags


def emit(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description='V10: tao TRAIN metadata, cach ly anh nghi van, KHONG train')
    root = Path(__file__).resolve().parent
    ap.add_argument('--manifest', default=str(root / 'dataset_manifest_SPLIT_DU_KIEN(1).csv'))
    ap.add_argument('--lock', default=str(root / 'EXTERNAL_SPLIT_LOCKED.csv'))
    ap.add_argument('--out', default=str(root / 'KET_QUA_V10'))
    args = ap.parse_args()
    manifest, lock, out = Path(args.manifest), Path(args.lock), Path(args.out)
    for p in (manifest, lock):
        if not p.is_file(): ap.error('Khong tim thay file: ' + str(p))
    out.mkdir(parents=True, exist_ok=True)

    old_yes, old_review = [], []
    counts = Counter(); classes = set(); old_provenance = defaultdict(Counter)
    by_old = defaultdict(Counter)
    old_paths = set()
    by_near = defaultdict(set)
    by_groups = defaultdict(set)
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        required = {'path', 'class', 'split', 'needs_label_review'}
        if not required.issubset(set(rd.fieldnames or [])):
            raise ValueError(f'Manifest thieu cot: {required-set(rd.fieldnames or [])}')
        for r in rd:
            counts['old_manifest_rows'] += 1
            path = r['path'].strip()
            cls = r['class'].strip()
            split = r['split'].strip().lower()
            review = (r.get('needs_label_review') or 'UNKNOWN').strip().upper()
            classes.add(cls)
            if split in ('train','val','test'):
                for key, bucket in (('near_group_id',by_near), ('group_id',by_groups)):
                    if r.get(key): bucket[r[key]].add(split)
            if split != 'train':
                continue
            if path.casefold() in old_paths:
                raise ValueError('Trung path TRAIN cu: '+path)
            old_paths.add(path.casefold())
            row = {'path': path, 'label': cls, 'origin': 'OLD97_MANIFEST',
                   'source_label_status': 'NOT_EXPERT_VERIFIED',
                   'needs_label_review': review, 'source': r.get('source',''),
                   'reason': '', 'split': 'train'}
            if review == 'NO':
                old_yes.append(row)
                by_old[cls]['unflagged'] += 1
            else:
                row['reason'] = 'OLD_MANIFEST_NEEDS_LABEL_REVIEW='+review
                old_review.append(row)
                by_old[cls]['flagged'] += 1
                folder = ntpath.dirname(path)
                old_provenance[cls][folder] += 1
    if len(classes) != 97: raise ValueError(f'Manifest khong dung 97 lop: {len(classes)}')
    bad_near = [x for x,v in by_near.items() if len(v)>1]
    bad_group = [x for x,v in by_groups.items() if len(v)>1]
    if bad_near or bad_group:
        raise ValueError(f'CO NGUY CO LEAK old groups: near {len(bad_near)}, group {len(bad_group)}')

    field_yes, field_review = [], []
    field_counts = Counter(); by_field = Counter(); quarantine_reasons = Counter()
    field_paths = set(); clusters=defaultdict(set); seen_paths=set()
    with lock.open(newline='', encoding='utf-8-sig') as f:
        rd = csv.DictReader(f)
        required = {'path','group','split','reason','crop','cluster'}
        if not required.issubset(set(rd.fieldnames or [])):
            raise ValueError(f'Lock thieu cot: {required-set(rd.fieldnames or [])}')
        for r in rd:
            path = r['path'].strip();group=r['group'].strip();split=r['split'].strip().lower()
            field_counts['split_'+split]+=1
            if path.casefold() in seen_paths: raise ValueError('Trung path locked: '+path)
            seen_paths.add(path.casefold())
            if split in ('train','val','test') and r['cluster'].strip():
                clusters[r['cluster'].strip()].add(split)
            if split != 'train': continue
            field_paths.add(path.casefold())
            label = GROUP_TO_LABEL.get(group, '')
            reasons=[]
            if r['reason'].strip() != 'OK': reasons.append('LOCK_REASON_NOT_OK')
            if group in UNVERIFIED_GROUPS: reasons.append('GROUP_LABEL_NOT_VERIFIED')
            elif not label: reasons.append('UNKNOWN_GROUP_LABEL')
            else: reasons.extend(conflict_flags(path, group))
            if path.casefold() in old_paths: reasons.append('OVERLAP_WITH_OLD_TRAIN')
            row = {'path':path,'label':label,'origin':'PLANTDOC_FIELD_LOCKED',
                   'source_label_status': 'SOURCE_GROUP_ONLY_NOT_EXPERT_VERIFIED',
                   'needs_label_review': 'YES' if reasons else 'SOURCE_GROUP_ONLY',
                   'source':group,'reason':';'.join(reasons),'split':'train'}
            if reasons:
                field_review.append(row)
                for reason in reasons: quarantine_reasons[reason]+=1
            else:
                field_yes.append(row)
                by_field[label] += 1
    leaks=[key for key, splits in clusters.items() if len(splits)>1]
    if leaks: raise ValueError('Phat hien cluster Field qua nhieu split, dung: '+str(len(leaks)))
    if not {'split_train','split_val','split_test'}.issubset(field_counts):
        raise ValueError('Khong tim thay du Train/Val/Test trong lock')
    if (len(field_yes)+len(field_review)) != field_counts['split_train']:
        raise ValueError('Khong duyet du FIELD Train')

    cols = ['path','label','origin','source_label_status','needs_label_review','source','reason','split']
    emit(out/'TRAIN_CANDIDATE_V10.csv', old_yes+field_yes, cols)
    emit(out/'TRAIN_QUARANTINE_V10.csv', old_review+field_review, cols)
    emit(out/'TRAIN_FIELD_SAFE_V10.csv', field_yes, cols)
    emit(out/'TRAIN_FIELD_CAN_XAC_MINH_V10.csv',field_review,cols)
    # Group old source folders for the 11 empty provisional classes.
    missing = sorted(c for c in classes if by_old[c]['unflagged']+by_field[c]==0)
    sources=[]
    for cls in missing:
        for folder,n in old_provenance[cls].most_common():
            sources.append({'class':cls,'source_folder':folder,'train_review_flagged':n,
                            'status':'CAN_XAC_MINH_NHAN_NGUON'})
    emit(out/'NGUON_11_LOP_CAN_XAC_MINH_V10.csv',sources,['class','source_folder','train_review_flagged','status'])
    coverage=[]
    for cls in sorted(classes):
        coverage.append({'class':cls,'old_train_unflagged':by_old[cls]['unflagged'],
                        'field_train_kept':by_field[cls],
                        'train_candidate':by_old[cls]['unflagged']+by_field[cls],
                        'old_train_needs_review':by_old[cls]['flagged'],
                        'train_class_missing': 'YES' if cls in missing else 'NO'})
    emit(out/'BAO_PHU_97_LOP_V10.csv',coverage,
         ['class','old_train_unflagged','field_train_kept','train_candidate',
          'old_train_needs_review','train_class_missing'])
    summary = {
        'mode':'TRAIN_ONLY_METADATA_QUARANTINE',
        'old_manifest_rows': counts['old_manifest_rows'],
        'old_train_unflagged':len(old_yes),
        'old_train_review':len(old_review),
        'field_split_metadata_counts':{s:field_counts['split_'+s] for s in ('train','val','test','exclude')},
        'field_train_candidate':len(field_yes),
        'field_train_review':len(field_review),
        'field_quarantine_reasons':dict(quarantine_reasons),
        'total_train_candidate':len(old_yes)+len(field_yes),
        'total_train_quarantine':len(old_review)+len(field_review),
        'classes_with_candidate_train':len(classes)-len(missing),
        'classes_without_candidate_train':missing,
        'old_cross_split_group_collisions':len(bad_group),
        'old_cross_split_near_group_collisions':len(bad_near),
        'field_cross_split_cluster_collisions':len(leaks),
        'no_val_or_test_images_read':True,
        'no_model_training':True,
        'no_original_label_modified':True,
        'important':'Unflagged/source group is not expert-verified. No new accuracy is claimed. TRAIN_CANDIDATE is provisional, not cleared for all 97 classes.'
    }
    with (out/'TONG_KET_V10.json').open('w',encoding='utf-8') as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('DA TAO TAI:',out)
    print('GUI LAI: TONG_KET_V10.json, TRAIN_FIELD_CAN_XAC_MINH_V10.csv, BAO_PHU_97_LOP_V10.csv')

if __name__ == '__main__': main()
