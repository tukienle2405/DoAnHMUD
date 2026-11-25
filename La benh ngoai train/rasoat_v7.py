# -*- coding: utf-8 -*-
"""V7 - Automatic triage of V6 misclassified plant-leaf images.
Runs locally on existing CSV/images. Never rewrites ground-truth labels or checkpoints.
Dependencies: Pillow (present in typical PyTorch/PIL project environments).
"""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import html as htmllib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageFilter, ImageStat
except ImportError:
    print("THIEU THU VIEN Pillow. Cai bang lenh: py -3 -m pip install Pillow")
    raise SystemExit(2)

ENC = 'utf-8-sig'
ROOT = Path(__file__).resolve().parent


def load_csv(path):
    with path.open('r',encoding=ENC,newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def find_html(custom):
    if custom:
        p=Path(custom)
        return p if p.exists() else None
    base_candidates = [
        ROOT/'KET_QUA_RASOAT_V6'/'XEM_ANH_DU_DOAN_SAI.html',
        ROOT/'XEM_ANH_DU_DOAN_SAI.html',
        ROOT.parent/'KET_QUA_RASOAT_V6'/'XEM_ANH_DU_DOAN_SAI.html',
        ROOT.parent/'XEM_ANH_DU_DOAN_SAI.html',
    ]
    return next((p for p in base_candidates if p.exists()),None)


def html_images(file):
    """Parse images in case order; useful when original D:\\ paths are not available."""
    if not file:
        return {}
    print('HTML anh:', file)
    text=file.read_text('utf-8', errors='replace')
    blocks=re.findall(r'<div\s+class="case"\s*>(.*?)(?=<div\s+class="case"\s*>|</body>)',text,flags=re.S|re.I)
    out={}
    for i, block in enumerate(blocks,1):
        m=re.search(r'<img\b[^>]*\bsrc=["\'](data:image/(?:jpeg|jpg|png|webp);base64,([^"\']+))["\']',block,re.S|re.I)
        if not m: continue
        try: out[i]=base64.b64decode(m.group(2),validate=False)
        except Exception: continue
    print('Anh duoc nhung trong HTML:',len(out),'/',len(blocks))
    return out


def normalize(text):
    return re.sub('[^a-z0-9]+','_',str(text).casefold()).strip('_')


def dhash(image):
    # 9x8 horizontal difference hash; used to generate review *candidates*, not remove images.
    small=image.convert('L').resize((9,8),Image.Resampling.LANCZOS)
    pixels=list(small.getdata());num=0
    for y in range(8):
        for x in range(8):
            num=(num<<1)|int(pixels[y*9+x]>pixels[y*9+x+1])
    return num


def stats_for_image(raw):
    with Image.open(io.BytesIO(raw)) as original:
        im=original.convert('RGB')
        w,height=im.size
        sample=im.copy(); sample.thumbnail((450,450))
        gray=sample.convert('L')
        sd=ImageStat.Stat(gray).stddev[0]
        # Edge-response variance is a *weak* visual quality heuristic (not a diagnosis).
        edge=gray.filter(ImageFilter.FIND_EDGES)
        edge_std=ImageStat.Stat(edge).stddev[0]
        bio=io.BytesIO();sample.save(bio,format='JPEG',quality=76)
        return {'width':w,'height':height,'contrast_std':round(sd,2),
                'edge_std':round(edge_std,2), 'sha256':hashlib.sha256(raw).hexdigest(),
                'dhash':dhash(sample),'thumbnail':base64.b64encode(bio.getvalue()).decode('ascii')}


def resolve_bytes(r, idx, html_data):
    p=Path(str(r.get('path','')).strip())
    if p.is_file():
        try: return p.read_bytes(),'file_goc'
        except OSError: pass
    if idx in html_data: return html_data[idx],'html_nhung'
    return None,'khong_tim_thay'


def issue_name(path,true_disease):
    stem=normalize(Path(path.replace('\\','/')).name)
    label=normalize(true_disease)
    problems=[]
    if ('bacterial_speck' in stem or 'bacterialspeck' in stem) and 'bacterial_spot' in label:
        problems.append('TEN_FILE_SPECK_NHAN_SPOT')
    if ('tobrfv' in stem or 'brown_rugose' in stem) and ('mosaic_virus' in label or 'mosaic' in label):
        problems.append('TEN_FILE_TOBRFV_NHAN_MOSAIC')
    if any(k in stem for k in ['diagram','infographic','poster','collage','chart','slides','screenshot','figure','illustration']):
        problems.append('TEN_FILE_CO_THE_LA_ANH_MINH_HOA')
    return problems


def build(args):
    csv_file=Path(args.csv) if args.csv else ROOT/'CAC_ANH_DU_DOAN_SAI_CAN_XEM.csv'
    if not csv_file.is_file():
        raise FileNotFoundError('Khong thay CSV: '+str(csv_file))
    rows=load_csv(csv_file)
    if not rows or 'true_disease' not in rows[0] or 'path' not in rows[0]:
        raise ValueError('CSV khong co cot true_disease/path; hay dung CSV V6')
    image_by_case=html_images(find_html(args.html))
    output=Path(args.outdir) if args.outdir else ROOT/'KET_QUA_V7_TU_DONG'
    output.mkdir(parents=True,exist_ok=True)
    print('Tong anh sai can ra soat:',len(rows))
    results=[];fingerprints=[];images={};counts=Counter(); by_disease=Counter(); by_crop=Counter()
    for idx,r in enumerate(rows,1):
        raw,source=resolve_bytes(r,idx,image_by_case)
        reasons=[]
        if str(r.get('model_disagreement','')).strip().lower() in ('1','true','yes'):
            reasons.append('HAI_MODEL_CAY_BAT_DONG')
        try: score=float(r.get('max_blend_probability') or 0)
        except (ValueError, TypeError): score=0
        if score>=0.8: reasons.append('DU_DOAN_SAI_DIEM_CAO')
        if normalize(r.get('true_crop',''))!=normalize(r.get('pred_crop','')):
            reasons.append('SAI_LOAI_CAY')
        else:
            reasons.append('DUNG_CAY_SAI_BENH')
        if 'healthy' in normalize(r.get('true_disease','')) and 'healthy' not in normalize(r.get('pred_disease','')):
            reasons.append('NHAN_HEALTHY_NHUNG_MODEL_BAO_BENH_CAN_XEM')
        reasons.extend(issue_name(str(r.get('path','')),str(r.get('true_disease',''))))
        meta={'width':'','height':'','contrast_std':'','edge_std':'','sha256':'','dhash':None}
        if raw is None:
            reasons.append('THIEU_ANH_GOC')
        else:
            try:
                meta=stats_for_image(raw)
                images[idx]=meta['thumbnail']
                fingerprints.append((idx,meta['sha256'],meta['dhash']))
                if min(meta['width'],meta['height'])<230: reasons.append('ANH_DO_PHAN_GIAI_THAP')
                if max(meta['width'],meta['height'])/max(min(meta['width'],meta['height']),1)>3.1:
                    reasons.append('ANH_TI_LE_BAT_THUONG')
                if meta['contrast_std']<22: reasons.append('ANH_DO_TUONG_PHAN_THAP')
                # Qualitative threshold: visual-review candidate only.
                if meta['edge_std']<17: reasons.append('CO_THE_MO_HOAC_NHIEU_NEN')
            except Exception:
                reasons.append('KHONG_DOC_DUOC_ANH')
        # Priority is triage logic, not a probability of label errors.
        weight={'TEN_FILE_SPECK_NHAN_SPOT':6,'TEN_FILE_TOBRFV_NHAN_MOSAIC':6,
                'THIEU_ANH_GOC':5,'HAI_MODEL_CAY_BAT_DONG':3,
                'SAI_LOAI_CAY':3,'DUNG_CAY_SAI_BENH':2,
                'DU_DOAN_SAI_DIEM_CAO':3,'NHAN_HEALTHY_NHUNG_MODEL_BAO_BENH_CAN_XEM':4,
                'ANH_DO_PHAN_GIAI_THAP':2,'ANH_TI_LE_BAT_THUONG':2,
                'ANH_DO_TUONG_PHAN_THAP':1,'CO_THE_MO_HOAC_NHIEU_NEN':1,
                'TEN_FILE_CO_THE_LA_ANH_MINH_HOA':2}
        priority_score=sum(weight.get(z,0) for z in reasons)
        item={'stt_html':idx,'path':r.get('path',''), 'true_crop':r.get('true_crop',''),
              'true_disease':r.get('true_disease',''),'pred_crop':r.get('pred_crop',''),
              'pred_disease':r.get('pred_disease',''),'model_score':round(score,5),
              'source_image':source,'width':meta['width'],'height':meta['height'],
              'priority_score':priority_score,'reasons':reasons,
              'nhan_da_duoc_chuyen_gia_xac_minh':'','ghi_chu_xac_minh':''}
        results.append(item)
        for z in reasons: counts[z]+=1
        by_disease[r.get('true_disease','')]+=1; by_crop[r.get('true_crop','')]+=1

    duplicate_groups=[]
    for i,(ix,sha,phash) in enumerate(fingerprints):
        for j in range(i+1,len(fingerprints)):
            jx,sha2,phash2=fingerprints[j]
            dist=(phash^phash2).bit_count()
            if sha==sha2:
                tag='TRUNG_BYTE_CHINH_XAC'
            elif dist<=5:
                tag='NGHI_GAN_TRUNG_CAN_XEM'
            else: continue
            duplicate_groups.append({'anh_1':ix,'anh_2':jx,'loai':tag,
                                     'dhash_distance':dist,
                                     'label_1':results[ix-1]['true_disease'],
                                     'label_2':results[jx-1]['true_disease']})
            for z in (results[ix-1],results[jx-1]):
                if tag not in z['reasons']:
                    z['reasons'].append(tag)
                    z['priority_score']+=5 if tag=='TRUNG_BYTE_CHINH_XAC' else 3
                    counts[tag]+=1
    results.sort(key=lambda x:(-x['priority_score'],-x['model_score'],x['stt_html']))
    out_rows=[]
    for x in results:
        y=x.copy();y['ly_do_can_ra_soat']='; '.join(x['reasons']);y.pop('reasons',None);out_rows.append(y)
    fields=['stt_html','priority_score','ly_do_can_ra_soat','true_crop','true_disease','pred_crop',
            'pred_disease','model_score','source_image','width','height','path',
            'nhan_da_duoc_chuyen_gia_xac_minh','ghi_chu_xac_minh']
    write_csv(output/'DANH_SACH_UU_TIEN_V7.csv',out_rows,fields)
    write_csv(output/'NHOM_ANH_GAN_TRUNG_V7.csv',duplicate_groups,
              ['anh_1','anh_2','loai','dhash_distance','label_1','label_2'])
    report={'so_anh_sai_v6':len(rows),'so_anh_co_the_xem':len(images),
            'so_anh_thieu':len(rows)-len(images),
            'so_cap_trung_hoac_gan_trung_can_xem':len(duplicate_groups),
            'so_luong_theo_co_hieu':dict(counts.most_common()),
            'anh_sai_theo_cay':dict(by_crop.most_common()),
            'anh_sai_theo_benh':dict(by_disease.most_common()),
            'top_20_stt_html':[x['stt_html'] for x in results[:20]],
            'luu_y':['CHI DANH DAU NGHI VAN, khong tu dong chuan doan hay sua nhan',
                    'Diem model khong phai do tin cay da hieu chuan',
                    'V6 la Validation da su dung nhieu lan, khong phai tap Test doc lap',
                    'Khong duoc dung anh Validation de fine-tune nhan/model.']}
    (output/'TONG_KET_V7.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    report_text=[
        'V7 - BAO CAO TU DONG RA SOAT LOI (KHONG SUA NHAN)',
        'Anh sai: '+str(len(rows)),'Anh doc duoc: '+str(len(images)),
        'Anh thieu/khong doc duoc: '+str(len(rows)-len(images)),
        'Cap trung/giong nhau CAN XEM LAI: '+str(len(duplicate_groups)),
        '', 'Nhung co hieu phat hien (co the trung nhau):'
    ]
    report_text.extend('- '+k+': '+str(v) for k,v in counts.most_common())
    report_text.extend(['','QUAN TRONG:',
                        '- Dung ket qua chi de xep hang anh can xem, KHONG tu sua nhan.',
                        '- Anh theo nhan healthy / du doan benh khong co nghia nhan healthy sai.',
                        '- Anh gan trung theo dHash co the la anh khac nhau; phai kiem chung.',
                        '- Test va checkpoint cu khong bi cham toi.',
                        '', 'FILE CAN GUI LAI: TONG_KET_V7.json, DANH_SACH_UU_TIEN_V7.csv, NHOM_ANH_GAN_TRUNG_V7.csv'])
    (output/'BAO_CAO_V7.txt').write_text('\n'.join(report_text),encoding='utf-8')
    # Standalone HTML preview for top review cases, with embedded thumbnails.
    previews=[]
    for z in results[:min(args.top,len(results))]:
        ix=z['stt_html']; reasons=', '.join(z['reasons'])
        pic=(f'<img loading="lazy" src="data:image/jpeg;base64,{images[ix]}">' if ix in images
             else '<p><em>Khong doc duoc anh</em></p>')
        previews.append('<article><b>#{}</b> <small>Uu tien: {}</small>{}<p><b>Goc:</b> {} / {}<br><b>Model:</b> {} / {}<br><b>Diem:</b> {:.3f}</p><p><b>Can xem:</b> {}</p><small>{}</small></article>'.format(
            ix,z['priority_score'],pic,*[htmllib.escape(str(z[k])) for k in ('true_crop','true_disease','pred_crop','pred_disease')],
            z['model_score'],htmllib.escape(reasons),htmllib.escape(str(z['path']))))
    html_doc='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>V7 - Anh uu tien</title><style>
    body{font:15px system-ui,Arial,sans-serif;background:#f8f9fa;color:#1b2733;max-width:1180px;margin:20px auto;padding:0 15px}
    main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}article{background:#fff;border:1px solid #dce2e6;border-radius:10px;padding:13px;overflow-wrap:anywhere}
    img{max-width:100%;width:100%;max-height:265px;object-fit:contain;background:#f4f4f4}small{color:#53606c}p{line-height:1.55}</style>
    <h1>V7 – Danh sách ảnh ưu tiên rà soát</h1><p>Tự động phân nhóm từ kết quả V6, KHÔNG kết luận bệnh hoặc sửa nhãn. Validation đã được dùng trước đây.</p><main>'''+ '\n'.join(previews)+'</main></html>'
    (output/'XEM_ANH_UU_TIEN_V7.html').write_text(html_doc,encoding='utf-8')
    print('===== XONG V7 =====')
    print('Thu muc ket qua:',output)
    print('Anh xem duoc:',len(images),'/',len(rows))
    print('Cap trung/gan trung can xem:',len(duplicate_groups))
    print('GỬI LẠI 3 FILE:')
    print(' - TONG_KET_V7.json')
    print(' - DANH_SACH_UU_TIEN_V7.csv')
    print(' - NHOM_ANH_GAN_TRUNG_V7.csv')


def main():
    ap=argparse.ArgumentParser(description='V7 - tu dong ra soat anh Validation bi du doan sai')
    ap.add_argument('--csv',help='Duong dan CSV anh sai V6')
    ap.add_argument('--html',help='HTML V6 chua anh nhung, tuy chon')
    ap.add_argument('--outdir',help='Thu muc ket qua')
    ap.add_argument('--top',type=int,default=50,help='So anh hang dau hien trong HTML')
    args=ap.parse_args()
    build(args)

if __name__=='__main__':
    try:main()
    except Exception as err:
        print('\n[LOI]',type(err).__name__+':',err)
        raise
