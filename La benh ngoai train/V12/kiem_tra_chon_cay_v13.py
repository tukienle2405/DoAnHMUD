# -*- coding: utf-8 -*-
"""V13: Compare auto-crop vs user-confirmed-crop simulation using V12B images.

This is an analysis/simulation ONLY. The crop from an old folder is NOT verified
and must not be treated as ground truth. No evaluation accuracy is computed.
No model training, checkpoint overwrite, source image edits, or Streamlit changes.
"""
from __future__ import annotations
import argparse
import base64
import csv
import html
import json
import sys
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / 'KET_QUA_QUET_ANH_V12' / 'DANH_SACH_UNG_VIEN_V12.csv'
V12C_CSV = HERE / 'KET_QUA_DU_DOAN_V12C' / 'DU_DOAN_ANH_V12C.csv'
RESULT = HERE / 'KET_QUA_CHON_CAY_V13'
V2 = Path(r'D:\DoanHMvUD\training_runs\field97_v2_safe\field97_best_candidate.pth')
V11 = HERE.parent / 'V11' / 'KET_QUA_V11' / 'V11_CANDIDATE_ONLY_NOT_DEPLOY.pth'


def csv_read(path: Path) -> list[dict]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_save(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def json_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def deps():
    try:
        import so_sanh_du_doan_v12c as legacy
        return legacy
    except ImportError as exc:
        raise RuntimeError('Khong tim thay so_sanh_du_doan_v12c.py trong thu muc V12. Giai nen bo V12C truoc.') from exc


def check(args, verbose=True):
    legacy=deps()
    status = {
        'mode':'CHECK_V13_READ_ONLY',
        'input_csv':str(args.csv), 'previous_predictions_csv':str(args.v12c),
        'checkpoint_v2_exists':args.v2.is_file(),
        'checkpoint_v11_exists':args.v11.is_file(),
        'old_folder_labels_verified':False,
        'no_accuracy_calculated':True,
        'no_original_images_modified':True,
        'no_training':True,
    }
    try:
        rows, valid, errors = legacy.load_data(args.csv)
        status['image_rows']=len(rows)
        status['images_found']=len(valid)
        status['image_errors']=errors[:15]
        old=csv_read(args.v12c)
        existing=[r.get('copied_relative_path','').replace('\\','/') for r in old]
        expected=[r.get('copied_relative_path','').replace('\\','/') for r in rows]
        status['previous_prediction_rows']=len(old)
        status['matched_image_keys']=len(set(expected)&set(existing))
        status['duplicate_previous_keys']=len(existing)-len(set(existing))
        status['ready_for_compare']=(len(valid)==len(rows)>0 and len(old)==len(rows)
            and set(expected)==set(existing) and len(expected)==len(set(expected))
            and len(existing)==len(set(existing)) and args.v2.is_file() and args.v11.is_file())
    except Exception as exc:
        status['ready_for_compare']=False
        status['error']=type(exc).__name__+': '+str(exc)
    json_save(RESULT/'KIEM_TRA_V13.json',status)
    if verbose: print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
    return status


def html_zip(rows:list[dict], root:Path, dest:Path, max_images=24):
    from PIL import Image, ImageOps
    legacy=deps()
    p=[]
    sorted_rows=sorted(rows,key=lambda r:(-int(r['uu_tien']),r['relative_path']))[:max_images]
    for r in sorted_rows:
        img=legacy.safe_image(root,r['relative_path'])
        with Image.open(img) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((460,330))
            b=BytesIO();im.save(b,format='JPEG',quality=77,optimize=True)
        src='data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode('ascii')
        e=lambda x:html.escape(str(x),quote=True)
        p.append('<article><div class="image"><img src="'+src+'" alt="Anh la"></div><div><b>'+e(r['relative_path'])+'</b>'
         +'<p>Goi y tu folder: <b>'+e(r['cay_thu_muc'])+'</b> ('+e(r['nhan_thu_muc'])+')<br>'
         +'V3 tu dong: <b>'+e(r['cay_tu_dong'])+'</b><br>'
         +'V2 theo V3: <b>'+e(r['v2_theo_v3'])+'</b><br>'
         +'V2 neu chon cay trong folder: <b>'+e(r['v2_theo_cay_chon'])+'</b><br>'
         +'V11 neu chon cay trong folder: <b>'+e(r['v11_theo_cay_chon'])+'</b><br>'
         +'Canh bao: '+e(r['canh_bao'])+'</p></div></article>')
    sheet=('<html lang="vi"><meta charset="utf-8"><title>V13 - Thu chon cay</title><style>'
       'body{font:15px system-ui;background:#f7f9fb;color:#182333;max-width:1100px;margin:auto;padding:14px}'
       'article{display:flex;gap:15px;background:#fff;padding:12px;margin:12px 0;border:1px solid #ddd;border-radius:10px}'
       'img{width:240px;height:190px;object-fit:contain}p{line-height:1.6}'
       '@media(max-width:600px){article{display:block}img{width:100%}}</style>'
       '<h2>V13 - So sanh chon cay voi tu dong</h2><p><b>QUAN TRONG:</b> Ten thu muc la GOI Y CHUA XAC MINH. '
       'Day la thu nghiem, khong phai accuracy, khong chuan doan benh. '
       'Trong app chi cho suy luan theo cay khi nguoi dung da xac nhan cay.</p>'
       +''.join(p)+'</html>')
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('XEM_V13.html',sheet)
        z.writestr('GHI_CHU.txt','V13 KHONG TINH ACCURACY. Nen xem ca nguon anh va ten cay; folder co the sai. KHONG tu dong sua nhan.\n')
    return len(sorted_rows)


def compare(args):
    state=check(args,verbose=False)
    if not state.get('ready_for_compare'):
        print(json.dumps(state,ensure_ascii=False,indent=2))
        raise RuntimeError('Du lieu/checkpoint chua san sang. Xem KET_QUA_CHON_CAY_V13/KIEM_TRA_V13.json')
    legacy=deps()
    if args.batch_size<1:raise ValueError('batch_size phai >= 1')
    import torch
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA khong kha dung. Dung --device cpu')
    device=torch.device('cuda' if (args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available())) else 'cpu')
    print('V13 device:',device,flush=True)
    m2,n2=legacy.construct_model(args.v2,97,device)
    m11,n11=legacy.construct_model(args.v11,97,device)
    if set(n2)!=set(n11):raise ValueError('97 nhan V2/V11 khong khop')
    data_rows, pairs, _ = legacy.load_data(args.csv)
    old={r['copied_relative_path'].replace('\\','/'):r for r in csv_read(args.v12c)}
    rows=[]
    with torch.inference_mode():
        for off in range(0,len(pairs),args.batch_size):
            group=pairs[off:off+args.batch_size]
            inputs=legacy.prepare_batch([p for _,p in group]).to(device)
            y2=m2(inputs).cpu(); y11=m11(inputs).cpu()
            for j,(src,imgpath) in enumerate(group):
                key=src['copied_relative_path'].replace('\\','/')
                prev=old[key]
                chosen_crop=src['crop'].strip()
                if chosen_crop not in set(legacy.crop_from_name(n) for n in n2):
                    raise ValueError('Cay khong co trong 97 nhan: '+chosen_crop)
                p2,s2,t2=legacy.within_crop_prob(y2[j],n2,chosen_crop)
                p11,s11,t11=legacy.within_crop_prob(y11[j],n11,chosen_crop)
                original_crop=prev['cay_du_doan_v3']
                warning=legacy.filename_warning(src)
                different=chosen_crop!=original_crop
                rows.append({
                    'relative_path':key, 'cay_thu_muc':chosen_crop,
                    'nhan_thu_muc':src.get('folder_label_hint',''),
                    'cay_tu_dong':original_crop,'top3_cay_tu_dong':prev.get('top3_cay_v3',''),
                    'v2_theo_v3':prev.get('benh_du_doan_v2',''),
                    'v11_theo_v3':prev.get('benh_du_doan_v11',''),
                    'v2_theo_cay_chon':p2, 'v2_top3_theo_cay_chon':t2, 'v2_softmax_trong_cay':s2,
                    'v11_theo_cay_chon':p11,'v11_top3_theo_cay_chon':t11,'v11_softmax_trong_cay':s11,
                    'cay_thu_muc_khac_v3':int(different),
                    'v2_benh_doi_khi_chon_cay':int(p2!=prev.get('benh_du_doan_v2','')),
                    'v11_benh_doi_khi_chon_cay':int(p11!=prev.get('benh_du_doan_v11','')),
                    'v2_v11_bat_dong_khi_chon_cay':int(p2!=p11),
                    'canh_bao':warning,
                    'uu_tien':int(different)*3+int(bool(warning))*4+int(p2!=p11)*2,
                    'da_xac_minh_loai_cay':'NO','da_xac_minh_benh':'NO','la_test_doc_lap':'NO',
                    'original_path':src.get('original_path','')
                })
            print('Da phan tich',min(off+args.batch_size,len(pairs)),'/',len(pairs),flush=True)
    csv_save(RESULT/'SO_SANH_CHON_CAY_V13.csv',rows)
    previews=html_zip(rows,args.csv.parent,RESULT/'ANH_CAN_XEM_V13.zip')
    per_crop=[]
    for crop in sorted(set(r['cay_thu_muc'] for r in rows)):
        g=[r for r in rows if r['cay_thu_muc']==crop]
        per_crop.append({'crop':crop,'total':len(g),
                         'v3_khac_folder':sum(r['cay_thu_muc_khac_v3'] for r in g),
                         'v2_changes_when_select':sum(r['v2_benh_doi_khi_chon_cay'] for r in g),
                         'v11_changes_when_select':sum(r['v11_benh_doi_khi_chon_cay'] for r in g)})
    summ={
        'mode':'V13_EXPLORATORY_SELECT_CROP_FROM_OLD_FOLDER_NOT_GROUND_TRUTH',
        'images':len(rows), 'different_crop_from_folder':sum(r['cay_thu_muc_khac_v3'] for r in rows),
        'v2_disease_prediction_changed_when_forcing_folder_crop':sum(r['v2_benh_doi_khi_chon_cay'] for r in rows),
        'v11_disease_prediction_changed_when_forcing_folder_crop':sum(r['v11_benh_doi_khi_chon_cay'] for r in rows),
        'v2_v11_disease_disagreement_under_folder_crop':sum(r['v2_v11_bat_dong_khi_chon_cay'] for r in rows),
        'preview_images':previews,'breakdown':per_crop,
        'verified_crop_labels':0,'verified_disease_labels':0,'accuracy_calculated':False,
        'training_performed':False,'checkpoints_overwritten':False,'streamlit_modified':False,
        'warnings':['Ten thu muc khong phai nhan da xac minh. Chung minh hoa cach cho phep nguoi dung CHON CAY, khong chung minh mo hinh tot hon.',
                    'Phai hoi nguoi dung xac nhan loai cay truoc khi suy luan benh theo cay da chon.',
                    'Softmax gioi han trong cay KHONG phai do tin cay da hieu chuan.',
                    'V13 khong phai tap test doc lap; khong su dung de bao cao Accuracy.'],
        'files_to_send':['TONG_KET_V13.json','SO_SANH_CHON_CAY_V13.csv','ANH_CAN_XEM_V13.zip']
    }
    json_save(RESULT/'TONG_KET_V13.json',summ)
    print('DA XONG V13\n'+json.dumps(summ,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description='V13: cho phep thu du doan benh theo loai cay duoc xac nhan (thu bang ten folder)')
    parser.add_argument('--mode',choices=['check','compare'],default='check')
    parser.add_argument('--csv',type=Path,default=INPUT_CSV)
    parser.add_argument('--v12c',type=Path,default=V12C_CSV)
    parser.add_argument('--v2',type=Path,default=V2)
    parser.add_argument('--v11',type=Path,default=V11)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    if args.mode=='check':
        if not check(args)['ready_for_compare']:sys.exit(2)
    else:compare(args)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('LOI V13: '+type(exc).__name__+': '+str(exc), file=sys.stderr,flush=True)
        sys.exit(2)
