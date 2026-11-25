# -*- coding: utf-8 -*-
"""V12C: Blind prediction comparison V3 crop + V2/V11 disease on V12B old local candidates.

Does NOT claim accuracy, does NOT infer ground truth from folder names,
does NOT touch Train/Val/Test, does NOT train or alter checkpoints.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html
import json
import os
import sys
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT_DIR = HERE / 'KET_QUA_QUET_ANH_V12'
DEFAULT_CSV = INPUT_DIR / 'DANH_SACH_UNG_VIEN_V12.csv'
OUTPUT_DIR = HERE / 'KET_QUA_DU_DOAN_V12C'
M18 = Path(r'D:\DoanHMvUD\training_runs\field18_v3_safe\field18_best_candidate_v3.pth')
M2 = Path(r'D:\DoanHMvUD\training_runs\field97_v2_safe\field97_best_candidate.pth')
M11 = ROOT / 'V11' / 'KET_QUA_V11' / 'V11_CANDIDATE_ONLY_NOT_DEPLOY.pth'


def read_rows(p):
    with p.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def save_json(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def save_csv(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def safe_image(base,relative):
    from pathlib import PurePosixPath
    rel=str(relative).replace('\\','/')
    p=PurePosixPath(rel)
    if not rel or p.is_absolute() or '..' in p.parts or ':' in rel:
        raise ValueError('Duong dan anh khong an toan: '+rel)
    resolved=(base / Path(*p.parts)).resolve()
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError('Anh nam ngoai thu muc output V12B: '+rel)
    return resolved


def load_data(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(f'Chua co file: {csv_path}\nHay chay V12B truoc hoac chi duong dan --csv')
    rows=read_rows(csv_path)
    if not rows:raise ValueError('CSV khong co anh duoc chon.')
    require={'crop','copied_relative_path','original_path','folder_label_hint'}
    missing=require-set(rows[0])
    if missing:raise ValueError('CSV V12B thieu cot: '+', '.join(sorted(missing)))
    good=[]; errors=[]; seen=set()
    for i,r in enumerate(rows,2):
        rel=r['copied_relative_path'].strip()
        if rel.casefold() in seen:
            errors.append(f'Dong {i}: anh bi liet ke trung: {rel}')
            continue
        seen.add(rel.casefold())
        try:
            img=safe_image(csv_path.parent,rel)
            if not img.is_file():
                errors.append(f'Dong {i}: khong tim thay anh: {img}')
                continue
            good.append((r,img))
        except (OSError, ValueError) as e:
            errors.append(f'Dong {i}: {e}')
    return rows,good,errors


def check(args):
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    state={'mode':'CHECK_NO_INFERENCE', 'input_csv':str(args.csv),
           'models':{'crop_v3':str(args.v3), 'disease_v2':str(args.v2), 'disease_v11':str(args.v11)},
           'checkpoint_present':{'crop_v3':args.v3.is_file(),'disease_v2':args.v2.is_file(),'disease_v11':args.v11.is_file()},
           'no_training':True, 'original_images_untouched':True,
           'warning':'Anh V12B la anh cu, nhan khong xac minh. KHONG duoc tinh Accuracy.'}
    try:
        rows, good, errors=load_data(args.csv)
        state.update({'csv_rows':len(rows),'images_found':len(good),'missing_or_invalid_images':errors[:20]})
        if len(rows)!=len(good):state['images_complete']=False
        else:state['images_complete']=True
    except Exception as e:
        state.update({'images_complete':False,'data_error':f'{type(e).__name__}: {e}'})
    state['ready_for_predict']=all(state['checkpoint_present'].values()) and state['images_complete']
    save_json(OUTPUT_DIR/'KIEM_TRA_V12C.json',state)
    print(json.dumps(state,ensure_ascii=False,indent=2),flush=True)
    return state


def crop_rank(model_logits, names):
    import torch
    scores=torch.softmax(model_logits,dim=-1)
    order=torch.argsort(scores,descending=True)[:3].tolist()
    return names[order[0]],round(float(scores[order[0]]),4),' | '.join(names[i] for i in order)


def within_crop_prob(logits,names,crop):
    import torch
    # Probability is relative only to the candidate diseases for the predicted crop.
    ids=[i for i,n in enumerate(names) if crop_from_name(n)==crop]
    if not ids:raise ValueError('Khong co benh cho cay du doan: '+crop)
    scores=torch.softmax(logits[ids],dim=0)
    top=torch.argsort(scores,descending=True)[:min(3,len(ids))]
    i=int(top[0])
    return names[ids[i]],round(float(scores[i]),4),' | '.join(names[ids[int(j)]] for j in top)


def crop_from_name(label):
    if label.startswith('Tomato_'):return 'Tomato'
    if label.startswith('Pepper__bell___'):return 'Pepper__bell'
    return label.split('___',1)[0]


def filename_warning(row):
    filename=Path(row.get('original_path','')).name.lower()
    crop=row['crop']
    if crop=='Tomato' and ('eggplant' in filename or 'soybean' in filename):
        return 'TEN_FILE_GOI_Y_CAY_KHAC'
    if crop=='Lemon' and 'grapefruit' in filename:
        return 'TEN_FILE_GOI_Y_CAY_KHAC'
    if crop=='Coffee' and 'rust' in filename and 'chay' in row.get('folder_label_hint','').lower():
        return 'TEN_FILE_VA_NHAN_THU_MUC_CO_THE_BAT_DONG'
    return ''


def create_preview(rows, input_root, output_path, top_n=24):
    # Zip contains a standalone HTML with embedded thumbnails of suspicious samples.
    from PIL import Image,ImageOps
    chosen=sorted(rows,key=lambda r:(-int(r['do_uu_tien']),r['copied_relative_path']))[:top_n]
    html_fragments=['<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>V12C - xem anh nghi van</title>',
      '<style>body{font:15px system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 14px;background:#f7f8fa;color:#243045}',
      '.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}',
      'article{background:white;border:1px solid #dadfe6;border-radius:10px;padding:12px;overflow-wrap:anywhere}',
      'img{width:100%;height:230px;object-fit:contain;background:#f1f2f4}p{line-height:1.5}</style></head><body>',
      '<h2>V12C · Anh can xem lai</h2><p>Du doan so sanh V2 va V11, chung model cay V3. KHONG phai chan doan benh; KHONG phai danh gia Accuracy. Nhan thu muc CHUA xac minh.</p>',
      '<div class="cards">']
    for r in chosen:
        path=safe_image(input_root,r['copied_relative_path'])
        try:
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((500,360))
                buff=BytesIO();im.save(buff,format='JPEG',quality=78)
                src='data:image/jpeg;base64,'+base64.b64encode(buff.getvalue()).decode('ascii')
        except Exception:
            src=''
        e=lambda x:html.escape(str(x),quote=True)
        html_fragments.extend([
            '<article><b>'+e(r['crop_thu_muc'])+'</b> · '+e(r['copied_relative_path'])+'<br>',
            '<img loading="lazy" src="'+src+'" alt="Anh la">',
            '<p><b>V3 đoán cây:</b> '+e(r['cay_du_doan_v3'])+'<br>',
            '<b>V2:</b> '+e(r['benh_du_doan_v2'])+'<br>',
            '<b>V11:</b> '+e(r['benh_du_doan_v11'])+'<br>',
            '<b>Gợi ý thư mục:</b> '+e(r['goi_y_nhan_thu_muc'])+'<br>',
            '<b>Cờ ưu tiên:</b> '+e(r['co_uu_tien'])+'</p>',
            '<small>'+e(r['original_path'])+'</small></article>'
        ])
    html_fragments.append('</div></body></html>')
    with zipfile.ZipFile(output_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        z.writestr('XEM_ANH_V12C.html',''.join(html_fragments))
        def buf_csv(s):
            b=BytesIO(); text=__import__('io').TextIOWrapper(b,encoding='utf-8-sig',newline='',write_through=True)
            w=csv.DictWriter(text,fieldnames=list(s[0]),extrasaction='ignore');w.writeheader();w.writerows(s)
            text.flush();return b.getvalue()
        if chosen:
            z.writestr('DANH_SACH_CAN_XEM_V12C.csv',buf_csv(chosen))
        z.writestr('LUU_Y.txt','Ảnh chỉ phục vụ rà soát lỗi và chất lượng dữ liệu; không có nhãn bệnh đã xác minh. Không đánh giá Accuracy.\n')
    return len(chosen)


def construct_model(path,n,device):
    import torch
    from model_architecture_v12c import get_mobilenet_v2_model
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(checkpoint,dict) or 'model_state_dict' not in checkpoint or 'class_names' not in checkpoint:
        raise ValueError('Checkpoint khong dung cau truc: '+str(path))
    names=list(checkpoint['class_names'])
    if len(names)!=n or len(set(names))!=n:
        raise ValueError('Checkpoint khong dung so nhan: '+str(path))
    model=get_mobilenet_v2_model(num_classes=n,pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    return model.to(device).eval(),names


def prepare_batch(paths):
    import torch
    from PIL import Image,ImageOps
    from torchvision import transforms
    transform=transforms.Compose([
        transforms.Resize((224,224)),transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    prepared=[]
    for path in paths:
        with Image.open(path) as im:
            im=ImageOps.exif_transpose(im).convert('RGB')
            prepared.append(transform(im))
    return torch.stack(prepared)


def predict(args):
    st=check(args)
    if not st['ready_for_predict']:
        raise RuntimeError('DUNG: thieu file/checkpoint hoac anh. Xem KIEM_TRA_V12C.json de biet chi tiet.')
    if args.batch_size<1:raise ValueError('--batch_size phai >=1')
    import torch
    device=torch.device('cuda' if (args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available())) else 'cpu')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Khong co CUDA; dung --device cpu')
    print('Thiet bi:',device,flush=True)
    torch.set_grad_enabled(False)
    m18,n18=construct_model(args.v3,18,device)
    m2,n2=construct_model(args.v2,97,device)
    m11,n11=construct_model(args.v11,97,device)
    if set(n2)!=set(n11):raise ValueError('V2 va V11 khong co cung 97 nhan')
    if set(n18)!={crop_from_name(n) for n in n2}:raise ValueError('V3 va V2 khong dong nhat 18 cay')
    _,pairs,_=load_data(args.csv)
    outputs=[]
    with torch.inference_mode():
        for begin in range(0,len(pairs),args.batch_size):
            batch=pairs[begin:begin+args.batch_size]
            inputs=prepare_batch([path for _,path in batch]).to(device)
            ycrop=m18(inputs).cpu(); y2=m2(inputs).cpu(); y11=m11(inputs).cpu()
            for i,(data,image_path) in enumerate(batch):
                pred_crop,prob_crop,top3crop=crop_rank(ycrop[i],n18)
                p2,s2,t2=within_crop_prob(y2[i],n2,pred_crop)
                p11,s11,t11=within_crop_prob(y11[i],n11,pred_crop)
                gl2,gl2_score,_=crop_rank(y2[i],n2)
                gl11,gl11_score,_=crop_rank(y11[i],n11)
                # Global 97 crop is NOT necessarily the same as V3 predicted crop.
                flags=[]
                if pred_crop!=data['crop']:flags.append('V3_KHAC_TEN_THU_MUC')
                if p2!=p11:flags.append('V2_V11_BAT_DONG_BENH')
                if crop_from_name(gl2)!=pred_crop:flags.append('V2_TU_DO_KHAC_CAY_V3')
                if crop_from_name(gl11)!=pred_crop:flags.append('V11_TU_DO_KHAC_CAY_V3')
                warning=filename_warning(data)
                if warning:flags.append(warning)
                prio= (4 if warning else 0)+(3 if pred_crop!=data['crop'] else 0)+(2 if p2!=p11 else 0)+ (1 if crop_from_name(gl2)!=pred_crop or crop_from_name(gl11)!=pred_crop else 0)
                outputs.append({
                    'copied_relative_path':data['copied_relative_path'],
                    'crop_thu_muc':data['crop'], 'goi_y_nhan_thu_muc':data['folder_label_hint'],
                    'original_path':data['original_path'],
                    'cay_du_doan_v3':pred_crop,'softmax_cay_v3':prob_crop,'top3_cay_v3':top3crop,
                    'v3_khop_thu_muc':int(pred_crop==data['crop']),
                    'benh_du_doan_v2':p2, 'softmax_v2_theo_cay':s2,'top3_v2_theo_cay':t2,
                    'benh_du_doan_v11':p11,'softmax_v11_theo_cay':s11,'top3_v11_theo_cay':t11,
                    'v2_v11_bat_dong':int(p2!=p11),
                    'v2_tu_do':gl2,'v11_tu_do':gl11,
                    'do_uu_tien':prio,'co_uu_tien':' | '.join(flags),
                    'nhan_benh_duoc_xac_minh':'NO', 'la_test_doc_lap':'NO'
                })
            print(f'Da du doan: {min(begin+args.batch_size,len(pairs))}/{len(pairs)} anh',flush=True)
    out=OUTPUT_DIR
    save_csv(out/'DU_DOAN_ANH_V12C.csv',outputs)
    flagged=[r for r in outputs if r['do_uu_tien']>0]
    save_csv(out/'CAN_XEM_LAI_V12C.csv',flagged)
    count_preview=create_preview(flagged if flagged else outputs,args.csv.parent,out/'ANH_CAN_XEM_V12C.zip',top_n=24)
    crop_counts=Counter(r['crop_thu_muc'] for r in outputs)
    report={
        'mode':'V12C_OLD_LOCAL_IMAGES_PREDICTION_ONLY_NOT_HOLDOUT',
        'total_images':len(outputs), 'crop_folders':dict(crop_counts),
        'crop_v3_agrees_with_folder_hint':sum(r['v3_khop_thu_muc'] for r in outputs),
        'v2_v11_disease_prediction_disagreement':sum(r['v2_v11_bat_dong'] for r in outputs),
        'priority_images':len(flagged),'preview_images':count_preview,
        'models':{'crop_v3':str(args.v3),'disease_v2':str(args.v2),'disease_v11':str(args.v11)},
        'labels_verified':0,'accuracy_calculated':False,'is_independent_test':False,
        'no_training':True,'checkpoints_unchanged':True,
        'warnings':['Du lieu lay tu thu muc cu, khong phai TEST doc lap.',
                    'Ten thu muc chi la goi y, KHONG phai ground truth; ti le khop thu muc KHONG phai Accuracy.',
                    'Softmax chua hieu chuan, khong la do tin cay lam sang.',
                    'Khong tu dong sua nhan hoac dua checkpoint V11 vao Streamlit.'],
        'files_to_send':['TONG_KET_V12C.json','DU_DOAN_ANH_V12C.csv','ANH_CAN_XEM_V12C.zip']
    }
    save_json(out/'TONG_KET_V12C.json',report)
    print('V12C HOAN TAT\n'+json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description='V12C: so sanh du doan 40 anh cu; KHONG tinh accuracy')
    parser.add_argument('--mode',choices=['check','predict'],default='check')
    parser.add_argument('--csv',type=Path,default=DEFAULT_CSV)
    parser.add_argument('--v3',type=Path,default=M18)
    parser.add_argument('--v2',type=Path,default=M2)
    parser.add_argument('--v11',type=Path,default=M11)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    if args.mode=='check':
        st=check(args)
        if not st['ready_for_predict']:sys.exit(2)
    else:predict(args)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'LOI V12C: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        sys.exit(2)
