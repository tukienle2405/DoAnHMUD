# -*- coding: utf-8 -*-
"""V12B: tu quet anh trong cac thu muc cay, copy MAU UNG VIEN vao thu muc rieng.

KHONG coi anh tu du lieu cu la independent holdout; khong sua nhan/ckpt;
KHONG dua vao ANH_MOI_DOC_LAP, khong tinh Accuracy.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import os
import random
import shutil
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
CROP_ALIASES = {
    'ca chua':'Tomato', 'tomato':'Tomato',
    'ca phe':'Coffee', 'coffee':'Coffee',
    'cao su':'Rubber', 'rubber':'Rubber',
    'chanh':'Lemon', 'lemon':'Lemon',
    'che':'Tea', 'tea':'Tea',
    'chuoi':'Banana', 'banana':'Banana',
    'dieu':'Cashew', 'cashew':'Cashew',
    'du du':'Papaya', 'papaya':'Papaya',
    'khoai tay':'Potato', 'potato':'Potato',
    'lua nuoc':'Rice', 'lua':'Rice', 'rice':'Rice',
    'mia':'Sugarcane', 'sugarcane':'Sugarcane',
    'mit':'Jackfruit', 'jackfruit':'Jackfruit',
    'ngo':'Corn', 'corn':'Corn',
    'oi':'Guava', 'guava':'Guava',
    'ot cay':'Chili', 'ot':'Chili', 'chili':'Chili',
    'ot chuong':'Pepper__bell', 'pepper bell':'Pepper__bell',
    'xoai':'Mango', 'mango':'Mango',
    'sau rieng':'Durian', 'durian':'Durian',
}
SPLIT_FOLDERS = {'train','training','test','testing','validation','val','valid'}


def ascii_key(s: str) -> str:
    s = unicodedata.normalize('NFKD', s.lower().replace('đ','d'))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.replace('_',' ').replace('-',' ')
    return ' '.join(s.split())


def norm_path(p: str | Path) -> str:
    return str(p).strip().replace('/','\\').replace('\\\\','\\').rstrip('\\').casefold()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def references(base:Path, manifest_opt, locked_opt):
    manifest_opts = [Path(manifest_opt)] if manifest_opt else [
        base/'V10'/'dataset_manifest_SPLIT_DU_KIEN(1).csv',
        base/'V9'/'dataset_manifest_SPLIT_DU_KIEN(1).csv',
        base/'dataset_manifest_SPLIT_DU_KIEN(1).csv',
        HERE/'DU_LIEU_DOI_CHIEU'/'dataset_manifest_SPLIT_DU_KIEN(1).csv'
    ]
    locked_opts = [Path(locked_opt)] if locked_opt else [
        base/'V10'/'EXTERNAL_SPLIT_LOCKED.csv',
        base/'EXTERNAL_SPLIT_LOCKED.csv',
        base/'V9'/'EXTERNAL_SPLIT_LOCKED.csv',
        HERE/'DU_LIEU_DOI_CHIEU'/'EXTERNAL_SPLIT_LOCKED.csv'
    ]
    files = []
    for variants in [manifest_opts,locked_opts]:
        chosen = next((p for p in variants if p.is_file()),None)
        if not chosen:
            raise FileNotFoundError('Khong tim thay 1 trong 2 CSV doi chieu manifest / locked. KHONG chon anh de tranh ro ri du lieu. Da thu: '+ ' ; '.join(map(str,variants)))
        files.append(chosen)
    used_paths = set()
    old_paths = []
    phashes = set()
    counts=[]
    for fi in files:
        with fi.open('r',encoding='utf-8-sig',newline='') as f:
            r=csv.DictReader(f)
            if not r.fieldnames or 'path' not in r.fieldnames:raise ValueError('CSV thieu cot path: '+str(fi))
            c=0
            for row in r:
                raw = (row.get('path') or '').strip()
                if not raw:continue
                n=norm_path(raw)
                if n not in used_paths:old_paths.append(Path(raw))
                used_paths.add(n)
                if row.get('phash'):
                    try:phashes.add(int(row['phash'],16))
                    except ValueError:pass
                c+=1
            counts.append(c)
    return files, used_paths, old_paths, phashes, counts


def find_crop_folders(root:Path):
    result=[]
    for p in sorted(root.iterdir(),key=lambda x:x.name.casefold()):
        if not p.is_dir() or p.is_symlink():continue
        crop = CROP_ALIASES.get(ascii_key(p.name))
        if crop:result.append((crop,p))
    return result


def scan_tree(crop_folder:Path, used_paths, counts):
    candidates=[]
    for dirname, dirs, files in os.walk(crop_folder,followlinks=False):
        dirs[:]=[d for d in dirs if not Path(dirname,d).is_symlink()]
        base=Path(dirname)
        parts=base.relative_to(crop_folder).parts
        if any(ascii_key(s) in SPLIT_FOLDERS for s in parts):
            counts['bo_qua_thu_muc_train_val_test'] += sum(Path(dirname,n).suffix.lower() in IMAGE_EXT for n in files)
            dirs.clear()
            continue
        for n in files:
            if Path(n).suffix.lower() not in IMAGE_EXT:continue
            file = base / n
            counts['anh_quet'] += 1
            if file.is_symlink():counts['lien_ket_anh_bo_qua']+=1;continue
            if norm_path(file) in used_paths:
                counts['da_co_trong_manifest_hoac_split']+=1;continue
            candidates.append(file)
    return candidates


def diverse_pool(paths, crop_folder, pool_size, seed):
    # Tron theo thu muc con: khong chi chon 1 benh / 1 nguon.
    rng=random.Random(seed)
    buckets=defaultdict(list)
    for p in paths:
        rel=p.relative_to(crop_folder)
        group = ascii_key(rel.parts[0]) if len(rel.parts)>1 else '(root)'
        buckets[group].append(p)
    keys=sorted(buckets)
    rng.shuffle(keys)
    for k in keys:
        buckets[k].sort(key=lambda p:p.as_posix().casefold())
        rng.shuffle(buckets[k])
    dq={k:deque(buckets[k]) for k in keys}
    chosen=[]
    while keys and len(chosen)<pool_size:
        for k in list(keys):
            if dq[k]:chosen.append(dq[k].popleft())
            if not dq[k]:keys.remove(k)
            if len(chosen)>=pool_size:break
    return chosen


def phash_image(path):
    # phash thong thuong 32x32, giu phan so sanh voi phash 64-bit trong EXTERNAL_SPLIT_LOCKED.
    from PIL import Image, ImageOps
    import numpy as np
    with Image.open(path) as im:
        im=ImageOps.exif_transpose(im).convert('L').resize((32,32),Image.Resampling.LANCZOS)
        x=np.asarray(im,dtype=np.float64)
    k=np.arange(32,dtype=np.float64)
    C=np.cos((2*k[None,:]+1)*k[:,None]*np.pi/64)
    C[0,:]*=np.sqrt(1/32)
    C[1:,:]*=np.sqrt(2/32)
    low=(C@x@C.T)[:8,:8]
    median=np.median(low)
    b=(low>median).reshape(-1)
    res=0
    for bit in b:res=(res<<1)|int(bit)
    return res


def inspect_image(p, min_px):
    from PIL import Image
    with Image.open(p) as im:
        im.verify()
    with Image.open(p) as im:
        w,h=im.size
    if min(w,h)<min_px:return False,(w,h),'DO_PHAN_GIAI_THAP'
    return True,(w,h),''


def output_preview(out, selected):
    content=['<!doctype html><html lang="vi"><meta charset="utf-8"><title>V12 - anh ung vien tu du lieu cu</title>',
        '<style>body{font:15px system-ui;max-width:1100px;margin:auto;padding:16px;background:#fafbfc}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}article{border:1px solid #ddd;padding:12px;border-radius:10px;background:white}img{width:100%;height:210px;object-fit:contain}small{overflow-wrap:anywhere}</style>',
        '<h2>V12B - Anh ung vien tu thu muc cu</h2><p><b>KHONG PHAI TEST DOC LAP.</b> Khong co nhan duoc chuyen sang trang thai da xac minh. Can kiem tra nguon, nhan va trung noi dung.</p><main>']
    for r in selected:
        rel=Path(r['copied_relative_path']).as_posix()
        content += [f'<article><b>{html.escape(r["crop"])}</b><img loading="lazy" src="{html.escape(rel,quote=True)}"><small>{html.escape(r["folder_label_hint"])}<p>{html.escape(r["original_path"])}</p></small></article>']
    content.append('</main></html>')
    (out/'XEM_ANH_UNG_VIEN_V12.html').write_text(''.join(content),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description='V12B: quet va copy anh ung vien tu thu muc cay - KHONG danh gia Accuracy')
    ap.add_argument('--root',type=Path,default=HERE.parent,help='Thu muc La benh ngoai train')
    ap.add_argument('--output',type=Path,default=HERE/'KET_QUA_QUET_ANH_V12')
    ap.add_argument('--manifest',type=Path,default=None)
    ap.add_argument('--locked',type=Path,default=None)
    ap.add_argument('--max-per-crop',type=int,default=10)
    ap.add_argument('--max-total-mb',type=int,default=500)
    ap.add_argument('--min-px',type=int,default=224)
    ap.add_argument('--no-copy',action='store_true',help='Chi kiem tra, khong copy')
    args=ap.parse_args()
    if args.max_per_crop<1 or args.max_per_crop>100:ap.error('--max-per-crop tu 1 den 100')
    if args.max_total_mb<1:ap.error('--max-total-mb phai >=1')
    try:
        import PIL
    except ImportError as e:
        raise RuntimeError('Thieu Pillow. Chay: py -3 -m pip install pillow') from e
    root=args.root.resolve(); out=args.output.resolve()
    if not root.is_dir():raise FileNotFoundError('Khong co thu muc nguon: '+str(root))
    if norm_path(out)==norm_path(root):raise ValueError('Output khong duoc trung thu muc nguon')
    out.mkdir(parents=True,exist_ok=True)
    print('DOC CSV KHOA VA MANIFEST (de loai bo anh da dung)...',flush=True)
    ref_files,used_paths,old_paths,locked_phashes,meta_counts=references(root,args.manifest,args.locked)
    folders=find_crop_folders(root)
    if not folders:raise ValueError('Khong tim thay thu muc cay o cap goc. Dung --root "D:\\DoanHMvUD\\La benh ngoai train"')
    print(f'Da doc {len(used_paths):,} path cu, tim thay {len(folders)} thu muc nguon.',flush=True)
    counts=Counter()
    pools=[]
    group_stats=defaultdict(Counter)
    for crop, folder in folders:
        cc=Counter()
        pool=scan_tree(folder,used_paths,cc)
        group_stats[crop].update(cc)
        counts.update(cc)
        group_stats[crop]['ung_vien_chua_co_path']+=len(pool)
        chosen=diverse_pool(pool,folder, max(20,args.max_per_crop*6),20261125)
        pools.append((crop,folder,chosen))
        print(f'  {folder.name} -> {crop}: {cc["anh_quet"]:,} anh, {len(pool):,} path chua thay trong CSV cu; duyet toi da {len(chosen)} anh.',flush=True)
    # Lap chi muc kich thuoc file CU de phat hien trung byte duoc copy sang ten / thu muc khac.
    wanted_sizes=set()
    for crop,folder,paths in pools:
        for p in paths:
            try:wanted_sizes.add(p.stat().st_size)
            except OSError:pass
    print('DOI CHIEU KICH THUOC VA SHA256 VOI DU LIEU CU (co the mat thoi gian)...',flush=True)
    old_by_size=defaultdict(list); missing_old=0
    for idx,p in enumerate(old_paths,1):
        try:
            st=p.stat()
            if st.st_size in wanted_sizes:old_by_size[st.st_size].append(p)
        except OSError:missing_old+=1
        if idx%50000==0:print(f'  Da doi chieu stat {idx:,}/{len(old_paths):,} anh cu...',flush=True)
    old_sha_cache={}; selected_shas=set(); selected_phashes=[]
    selected=[]; skipped=Counter(); copied_bytes=0; no_phash=0
    max_bytes=args.max_total_mb*1024*1024
    image_out=out/'ANH_UNG_VIEN_TU_DU_LIEU_CU'
    for crop,folder,paths in pools:
        taken=0
        for p in paths:
            if taken>=args.max_per_crop:break
            try:
                size=p.stat().st_size
                if copied_bytes+size>max_bytes:skipped['VUOT_GIOI_HAN_DUNG_LUONG']+=1;continue
                ok,dims,why=inspect_image(p,args.min_px)
                if not ok:skipped[why]+=1;continue
                digest=sha256(p)
                if digest in selected_shas:
                    skipped['TRUNG_SHA256_GIUA_UNG_VIEN']+=1;continue
                # Hash toi da cac anh CU cung kich thuoc voi anh ung vien.
                old_collision=False
                for old in old_by_size.get(size,[]):
                    key=norm_path(old)
                    if key not in old_sha_cache:
                        try:old_sha_cache[key]=sha256(old)
                        except OSError:old_sha_cache[key]=None
                    if old_sha_cache[key]==digest:
                        old_collision=True;break
                if old_collision:skipped['TRUNG_SHA256_VOI_ANH_CU']+=1;continue
                try:
                    ph=phash_image(p)
                    if locked_phashes and min((ph^b).bit_count() for b in locked_phashes)<=4:
                        skipped['NGHI_TRUNG_PHASH_PLANTDOC_DA_KHOA']+=1;continue
                    if any((ph^b).bit_count()<=4 for b in selected_phashes):
                        skipped['NGHI_GAN_TRUNG_GIUA_UNG_VIEN']+=1;continue
                except (ImportError, OSError, ValueError) as e:
                    ph=None;no_phash+=1
                name='{:04d}_{}{}'.format(taken+1,digest[:12],p.suffix.lower())
                dest=image_out/crop/name
                if not args.no_copy:
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists():
                        if sha256(dest)!=digest:raise RuntimeError('Ten dich xung dot; tu choi ghi de: '+str(dest))
                    else:shutil.copy2(p,dest)
                selected_shas.add(digest)
                if ph is not None:selected_phashes.append(ph)
                source_subfolder=p.parent.relative_to(folder).as_posix()
                selected.append({'crop':crop,'source_crop_folder':folder.name,
                    'folder_label_hint':source_subfolder,'original_path':str(p),
                    'copied_relative_path':dest.relative_to(out).as_posix(),
                    'sha256':digest,'width':dims[0],'height':dims[1],'size_bytes':size,
                    'label_verified':'NO','true_class':'','source_status':'OLD_LOCAL_DATA_NOT_IN_PATH_LIST',
                    'independent_test':'NO_UNVERIFIED'})
                copied_bytes+=size;taken+=1
            except (OSError,ValueError) as e:
                skipped['LOI_DOC_ANH']+=1
        group_stats[crop]['anh_duoc_chon']+=taken
    sel_fields=['crop','source_crop_folder','folder_label_hint','original_path','copied_relative_path',
        'sha256','width','height','size_bytes','label_verified','true_class','source_status','independent_test']
    write_csv(out/'DANH_SACH_UNG_VIEN_V12.csv',selected,sel_fields)
    write_csv(out/'BANG_GAN_NHAN_UNG_VIEN_V12.csv',[
        {'copied_relative_path':r['copied_relative_path'], 'crop':r['crop'],
         'true_class':'','label_verified':'NO','source':'OLD_LOCAL_DATA',
         'source_id':'','collection_date':'','notes':'Can xac minh nguon, nhan, trung/gan trung voi tap cu'}
        for r in selected],['copied_relative_path','crop','true_class','label_verified','source','source_id','collection_date','notes'])
    write_csv(out/'SO_LUONG_THEO_CAY_V12.csv',[
        {'crop':crop,**dict(group_stats[crop])} for crop in sorted(group_stats)
    ], ['crop','anh_quet','da_co_trong_manifest_hoac_split','bo_qua_thu_muc_train_val_test','ung_vien_chua_co_path','anh_duoc_chon'])
    output_preview(out,selected)
    # Zip nho, thu toi da 2 hinh moi cay de gui lai AI phan tich; khong can gui hang tram anh goc.
    zip_count=0
    if not args.no_copy:
        sample=defaultdict(int)
        with zipfile.ZipFile(out/'ANH_MAU_GUI_CHATGPT_V12.zip','w',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
            for row in selected:
                crop=row['crop']
                if sample[crop]>=2:continue
                src=out/row['copied_relative_path']
                try:
                    from PIL import Image, ImageOps
                    from io import BytesIO
                    with Image.open(src) as im:
                        rgb=ImageOps.exif_transpose(im).convert('RGB')
                        rgb.thumbnail((640,640))
                        b=BytesIO();rgb.save(b,format='JPEG',quality=75,optimize=True)
                    z.writestr(f'{crop}/{Path(src).stem}.jpg',b.getvalue())
                    sample[crop]+=1;zip_count+=1
                except Exception:
                    skipped['KHONG_TAO_DUOC_ANH_PREVIEW']+=1
    info={
        'mode':'AUTO_SCAN_LOCAL_OLD_IMAGES_NOT_INDEPENDENT_TEST',
        'base_dir':str(root), 'reference_files':[str(p) for p in ref_files],
        'reference_rows':meta_counts,'reference_unique_paths':len(used_paths),
        'reference_local_files_missing':missing_old,
        'source_crop_folders':[{ 'crop':crop,'folder':str(folder)} for crop,folder in folders],
        'crop_folders_found':len(folders), 'images_scanned':counts['anh_quet'],
        'images_with_paths_absent_from_old_csv':sum(g['ung_vien_chua_co_path'] for g in group_stats.values()),
        'already_listed_in_old_metadata':counts['da_co_trong_manifest_hoac_split'],
        'images_selected':len(selected),'selected_by_crop':dict(Counter(r['crop'] for r in selected)),
        'copied':not args.no_copy, 'bytes_selected':copied_bytes,
        'known_old_files_sha_checked':len(old_sha_cache),
        'selected_phash_unavailable':no_phash,
        'skipped_selection':dict(skipped),
        'number_small_preview_images':zip_count,
        'no_training':True,'no_original_images_modified':True,
        'not_independent_holdout':True,'auto_labels_verified':False,
        'limitations':[
            'Anh tu folder cu khong phai tap test doc lap du da chua co path trong manifest.',
            'SHA256 doi chieu noi dung VOI file cu cung dung luong co the doc duoc, khong bao gom file cu thieu tren may.',
            'pHash so sanh voi PlantDoc khoa va giua anh duoc chon; khong bao phu toan bo 125k anh cu.',
            'Ten folder chi la goi y loai cay; KHONG duoc tu coi la nhan benh da xac minh.',
            'Anh trong thu muc ten train/test/val/validation duoc bo qua de tranh dung nham.'
        ]}
    write_json(out/'TONG_KET_QUET_ANH_V12.json',info)
    report=(f'V12B - TU DONG LAY ANH TU THU MUC CAY (KHONG PHAI TEST DOC LAP)\n\n'
            f'Quet {len(folders)} thu muc cay, {counts["anh_quet"]:,} anh.\n'
            f'Path chua co trong metadata cu: {info["images_with_paths_absent_from_old_csv"]:,}.\n'
            f'Chon {len(selected)} anh ung vien, dung luong {copied_bytes/1024/1024:.1f} MB.\n'
            f'Da so sanh SHA256 voi {len(old_sha_cache):,} file cu co cung kich thuoc de doc.\n'
            'Khong auto doi nhan, khong tinh Accuracy, khong mo test doc lap moi.\n\n'
            'GUI CHATGPT: TONG_KET_QUET_ANH_V12.json, DANH_SACH_UNG_VIEN_V12.csv, ANH_MAU_GUI_CHATGPT_V12.zip.\n')
    (out/'BAO_CAO_V12.txt').write_text(report,encoding='utf-8')
    print('\n'+report,flush=True)
    print('KET QUA O:',out,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(f'LOI: {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise
