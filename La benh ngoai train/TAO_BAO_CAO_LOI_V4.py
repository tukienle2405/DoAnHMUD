# -*- coding: utf-8 -*-
"""Optional: re-run V4 CSV error summaries with standard library only."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json

HERE=Path(__file__).resolve().parent
INPUT=HERE/'DU_DOAN_CHI_TIET_V4.csv'
OUT=HERE/'BAO_CAO_LOI_V4'

def main():
    rows=list(csv.DictReader(INPUT.open(encoding='utf-8-sig',newline='')))
    if len(rows)!=238:
        print('Canh bao: so luong anh khac 238:',len(rows))
    labeled=[r for r in rows if r['true_disease'].strip()]
    OUT.mkdir(exist_ok=True)
    report=[]; confusions=[]; samples=[]
    by_disease=defaultdict(list)
    for r in labeled:
        by_disease[r['true_disease']].append(r)
    for name, data in sorted(by_disease.items()):
        old=sum(int(r['old_joint_ok']) for r in data)
        new=sum(int(r['new_joint_ok']) for r in data)
        errors=Counter(r['new_disease'] for r in data if not int(r['new_joint_ok']))
        report.append({'true_disease':name,'total':len(data),'old_joint_correct':old,'new_joint_correct':new,
                       'delta_correct':new-old,'new_joint_pct':round(new/len(data)*100,2),
                       'main_wrong_label':errors.most_common(1)[0][0] if errors else '',
                       'main_wrong_count':errors.most_common(1)[0][1] if errors else 0})
        for guess,number in errors.most_common():
            confusions.append({'true_disease':name,'predicted_disease':guess,'count':number})
    for r in labeled:
        if not int(r['new_joint_ok']):
            samples.append({'path':r['path'],'true_crop':r['true_crop'],'true_disease':r['true_disease'],
                            'old_crop':r['old_crop'],'new_crop':r['new_crop'],'new_disease':r['new_disease'],
                            'error_type':'SAI_CAY' if not int(r['new_crop_ok']) else 'DUNG_CAY_SAI_BENH'})
    def save(path,items):
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    save(OUT/'DO_CHINH_XAC_THEO_BENH_V4.csv',report)
    save(OUT/'CAC_CAP_NHAM_LAN_V4.csv',confusions)
    save(OUT/'ANH_CAN_XEM_LAI_V4.csv',samples)
    overview={'all_images':len(rows),'labeled_images':len(labeled),
              'old_joint_correct':sum(int(r['old_joint_ok']) for r in labeled),
              'new_joint_correct':sum(int(r['new_joint_ok']) for r in labeled),
              'new_crop_correct_labeled':sum(int(r['new_crop_ok']) for r in labeled),
              'new_crop_wrong_labeled':sum(not int(r['new_crop_ok']) for r in labeled),
              'correct_crop_wrong_disease':sum(int(r['new_crop_ok']) and not int(r['new_joint_ok']) for r in labeled),
              'old_only_correct':sum(int(r['old_joint_ok']) and not int(r['new_joint_ok']) for r in labeled),
              'new_only_correct':sum(int(r['new_joint_ok']) and not int(r['old_joint_ok']) for r in labeled),
              'either_model_correct':sum(int(r['old_joint_ok']) or int(r['new_joint_ok']) for r in labeled),
              'note':'Based on previously examined locked PlantDoc Test, for error analysis only. Never tune V5 parameters from this report.'}
    (OUT/'TONG_HOP_V4.json').write_text(json.dumps(overview,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(overview,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
