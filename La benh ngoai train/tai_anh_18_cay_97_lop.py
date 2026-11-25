# -*- coding: utf-8 -*-
"""Tuyển ảnh lá ngoài PlantVillage cho chính xác 18 cây / 97 nhãn AI.

- Tự tải: ảnh có nhãn nguồn PlantDoc, Mendeley (nếu API công khai hoạt động),
  ảnh được biên tập từ trang nghiên cứu/khuyến nông, ảnh ứng viên Wikimedia Commons.
- KHÔNG biến ảnh tìm theo từ khóa thành nhãn bệnh chắc chắn.
- KHÔNG đụng vào dữ liệu hoặc model cũ.
- Xuất CSV đủ/thiếu 97 lớp và HTML để xem ảnh ngay trên máy.

Chạy: python tai_anh_18_cay_97_lop.py --fast
      python tai_anh_18_cay_97_lop.py --mendeley
      python tai_anh_18_cay_97_lop.py --report
      python tai_anh_18_cay_97_lop.py --check-duplicates
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import io
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:\DoanHMvUD") if os.name == 'nt' else HERE
ROOT = ROOT if ROOT.exists() else HERE
OUT = ROOT / 'external_field_pool_v2'
CACHE = OUT / '_cache'
INDEX = OUT / 'DANH_SACH_ANH.csv'
EVENTS = OUT / 'LOI_VA_SU_KIEN.csv'
CLASS_CSV = HERE / 'DANH_SACH_CHUAN_18_CAY_97_LOP.csv'
LEGACY = HERE / 'anh_nguon_tuyen_chon.csv'
EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
HTTP_HEADERS = {'User-Agent': 'FieldLeafDatasetCollector/2.0 (educational research; https://www.wikipedia.org)',
                'Accept': '*/*'}
FIELDS = ['class','crop','condition','bucket','file','source','source_label','source_url',
          'license','width','height','sha256','status','note']

# Nhãn của bộ PlantDoc: giữ nguyên tên bệnh, không tự đánh đồng bệnh khác tên.
PLANTDOC = {
    'tomato early blight leaf':'Tomato_Early_blight',
    'tomato leaf early blight':'Tomato_Early_blight',
    'tomato late blight leaf':'Tomato_Late_blight',
    'tomato leaf late blight':'Tomato_Late_blight',
    'tomato septoria leaf spot':'Tomato_Septoria_leaf_spot',
    'tomato leaf bacterial spot':'Tomato_Bacterial_spot',
    'tomato mold leaf':'Tomato_Leaf_Mold',
    'tomato leaf mold':'Tomato_Leaf_Mold',
    'tomato leaf mosaic virus':'Tomato__Tomato_mosaic_virus',
    'tomato mosaic virus leaf':'Tomato__Tomato_mosaic_virus',
    'tomato two spotted spider mites leaf':'Tomato_Spider_mites_Two_spotted_spider_mite',
    'tomato target spot leaf':'Tomato__Target_Spot',
    'potato leaf early blight':'Potato___Early_blight',
    'potato leaf late blight':'Potato___Late_blight',
    'corn gray leaf spot':'Corn___Gray_leaf_spot',
    'corn leaf blight':'Corn___Blight',
    'corn rust leaf':'Corn___Common_rust',
    # PlantDoc 'Tomato leaf'/'Potato leaf' does NOT prove healthy;
    # Bell pepper 'leaf spot' does NOT prove bacterial spot.
}

MENDELEY_SOURCES = [
    # Dữ liệu riêng, ưu tiên phiên bản nguyên bản, chụp ngoài đồng/vườn.
    ('94srbmydkw',1,'Rangamati_Mango_Jackfruit','https://data.mendeley.com/datasets/94srbmydkw/1'),
    ('82vn9k44ky',1,'Guava_Mango_Jackfruit_Field','https://data.mendeley.com/datasets/82vn9k44ky/1'),
    ('mfpxg4y65r',1,'Coffee_Peru_Field','https://data.mendeley.com/datasets/mfpxg4y65r/1'),
    ('5nfjzntwd8',1,'Banana_Field_Additional','https://data.mendeley.com/datasets/5nfjzntwd8/1'),
    ('44p8v6ywsm',1,'Papaya_Orchards_Original','https://data.mendeley.com/datasets/44p8v6ywsm/1'),
    ('86s4jzj2m4',1,'Rice_Field_Bangladesh','https://data.mendeley.com/datasets/86s4jzj2m4/1'),
    ('4sdw6gstgs',1,'Rice_2026_Diseases','https://data.mendeley.com/datasets/4sdw6gstgs/1'),
    ('mwbwbdzz7z',1,'Potato_Field_2026','https://data.mendeley.com/datasets/mwbwbdzz7z/1'),
    ('354fsxwccb',1,'Potato_Tomato_Field','https://data.mendeley.com/datasets/354fsxwccb/1'),
    ('rrff4vb35m',2,'Tea_Real_Life_Leaves','https://data.mendeley.com/datasets/rrff4vb35m/2'),
    ('ng85zg4xm4',1,'Guava_Field_2026','https://data.mendeley.com/datasets/ng85zg4xm4/1'),
    ('643f5bbc2t',5,'Lemon_Field_Alternative','https://data.mendeley.com/datasets/643f5bbc2t/5'),
    ('xx94td2tyt',1,'Cashew_Field_2025','https://data.mendeley.com/datasets/xx94td2tyt/1'),
    ('rzh99cj2rj',1,'Sugarcane_Field_2026','https://data.mendeley.com/datasets/rzh99cj2rj/1'),
    ('w9drvtxg9g',1,'BellPepper_New_2026','https://data.mendeley.com/datasets/w9drvtxg9g/1'),
    ('wzc6r6w5w5',3,'Chili_Field_2026','https://data.mendeley.com/datasets/wzc6r6w5w5/3'),
    ('mhjwyb5p48',1,'Durian_VinhLong_RealField','https://data.mendeley.com/datasets/mhjwyb5p48/1'),
]

# Cac bo NGUOI DUNG DA CO TRONG TRAIN: tuyet doi khong tai lan thu hai lam Test:
# Banana hzttvh774g/1-2 1773 anh; Coffee k36wnd6knb/1 3322 anh;
# Papaya 3kwgxg4stb/1; Lemon jkd6t3w635/1; Rubber 4kjz78m7x5/2;
# Sugarcane 355y629ynj/1; Rice dwtn3c6w6p/1; Guava fspx44mwfp/1;
# Durian pxzvksbwnj/3-5; Tea 94fzcdz8gz/1.

# Bo 2595 anh Durian Leaf Diseases duoc nguoi dung dua vao Train:
# KHONG tai bo durian pxzvksbwnj vao external_test.

# An toan: chi chap nhan nhan Mendeley neu co ten benh khop han.
FIELD_MAP = {
  'Mango':{
    'anthracnose':'Mango___Anthracnose',
    'dieback':'Mango___Die_back','die back':'Mango___Die_back',
    'healthy':'Mango___healthy','healthy leaf':'Mango___healthy',
    'powdery mildew':'Mango___Powdery_mildew',
    'sooty mould':'Mango___Sooty_mould','sooty mold':'Mango___Sooty_mould',
  },
  'Jackfruit':{
    'algal leaf spot':'Jackfruit___Algal','jackfruit algal':'Jackfruit___Algal',
    'healthy leaf':'Jackfruit___healthy','healthy':'Jackfruit___healthy',
    'leaf spot':'Jackfruit___Leaf_spot',
    'sooty mold':'Jackfruit___Sooty_mould','sooty mould':'Jackfruit___Sooty_mould',
    'anthracnose':'Jackfruit___Anthracnose',
  },
  'Guava':{'healthy':'Guava___healthy','healthy leaf':'Guava___healthy',
           'anthracnose':'Guava___Anthracnose','red rust':'Guava___Rust',
           'rust':'Guava___Rust'},
  'Coffee':{'healthy':'Coffee___healthy','healthy leaf':'Coffee___healthy',
            'leaf rust':'Coffee___Rust','coffee rust':'Coffee___Rust','rust':'Coffee___Rust'},
  'Banana':{'healthy banana leaf':'Banana___healthy','healthy leaf':'Banana___healthy',
            'healthy':'Banana___healthy','sigatoka':'Banana___Sigatoka',
            'cordana':'Banana___Cordana'},
  'Papaya':{'healthy':'Papaya___healthy','healthy leaf':'Papaya___healthy',
            'anthracnose':'Papaya___Anthracnose','mealybug':'Papaya___Mealybug',
            'mites':'Papaya___Mite_damage','mite':'Papaya___Mite_damage',
            'mosaic':'Papaya___Mosaic','ringspot':'Papaya___Ringspot','ring spot':'Papaya___Ringspot',
            'leaf curl':'Papaya___Curl','curl':'Papaya___Curl'},
  'Rice':{'brown spot':'Rice___Brown_spot','leaf smut':'Rice___Leaf_smut',
          'bacterial leaf blight':'Rice___Bacterial_leaf_blight'},
  'Potato':{'early blight':'Potato___Early_blight','late blight':'Potato___Late_blight',
            'fungal late blight':'Potato___Late_blight',
            'healthy':'Potato___healthy','healthy leaf':'Potato___healthy'},
  'Tomato':{'early blight':'Tomato_Early_blight','late blight':'Tomato_Late_blight',
            'healthy':'Tomato_healthy','healthy leaf':'Tomato_healthy'},
  'Tea':{'healthy':'Tea___healthy','brown blight':'Tea___Brown_blight',
         'red leaf spot':None,'algal leaf spot':None,'grey blight':None},
  'Lemon':{'anthracnose':'Lemon___Anthracnose','bacterial blight':'Lemon___Bacterial_blight',
           'citrus canker':'Lemon___Citrus_canker','curl leaf':'Lemon___Curl_virus',
           'curl virus':'Lemon___Curl_virus','deficiency':'Lemon___Deficiency_leaf',
           'dry leaf':'Lemon___Dry_leaf','sooty mould':'Lemon___Sooty_mould',
           'spider mites':'Lemon___Spider_mites','healthy':'Lemon___healthy'},
  'Cashew':{'healthy':'Cashew___healthy','healthy leaf':'Cashew___healthy'},
  'Pepper__bell':{'healthy':'Pepper__bell___healthy','healthy leaf':'Pepper__bell___healthy',
                   'bacterial spot':'Pepper__bell___Bacterial_spot'},
  'Chili':{'healthy':'Chili___healthy','healthy leaf':'Chili___healthy',
           'fresh leaf':'Chili___healthy',
           'cercospora leaf spot':'Chili___Cercospora_leaf_spot'},
  'Sugarcane':{'pokkah boeng':'Sugarcane___Pokkah_boeng','pokkahboeng':'Sugarcane___Pokkah_boeng',
               'brown spot':'Sugarcane___Brown_spot','brown rust':'Sugarcane___Brown_rust',
               'smut':'Sugarcane___Smut','healthy':'Sugarcane___healthy'},
  'Durian':{},  # Bo 4000 anh chua nhieu anh than/qua/re; khong ep vao benh la AI
}

# Mot so bo Mendeley khong ghi ten cay o cap folder, nen ap scope ro rang.
SOURCE_CROP = {
 'Rangamati_Mango_Jackfruit':None,'Guava_Mango_Jackfruit_Field':None,
 'Coffee_Peru_Field':'Coffee','Banana_Field_Additional':'Banana',
 'Papaya_Orchards_Original':'Papaya','Rice_Field_Bangladesh':'Rice',
 'Rice_2026_Diseases':'Rice','Potato_Field_2026':'Potato',
 'Potato_Tomato_Field':None,'Tea_Real_Life_Leaves':'Tea',
 'Guava_Field_2026':'Guava','Lemon_Field_Alternative':'Lemon',
 'Cashew_Field_2025':'Cashew','Sugarcane_Field_2026':'Sugarcane',
 'BellPepper_New_2026':'Pepper__bell','Chili_Field_2026':'Chili',
 'Durian_VinhLong_RealField':'Durian'}

CROP_SYNONYMS = {
 'Banana':['banana','musa'],'Cashew':['cashew','anacardium'],'Chili':['chili','chilli','capsicum'],
 'Coffee':['coffee','coffea'],'Corn':['corn','maize','zea mays'],'Durian':['durian','durio'],
 'Guava':['guava','psidium'],'Jackfruit':['jackfruit','artocarpus'],'Lemon':['lemon','citrus limon'],
 'Mango':['mango','mangifera'],'Papaya':['papaya','carica papaya'],
 'Pepper__bell':['bell pepper','capsicum annuum'],'Potato':['potato','solanum tuberosum'],
 'Rice':['rice','oryza sativa'],'Rubber':['rubber tree','hevea'],
 'Sugarcane':['sugarcane','sugar cane','saccharum'],'Tea':['tea leaf','tea plant','camellia sinensis'],
 'Tomato':['tomato','solanum lycopersicum']}


def norm(s):
    return ' '.join(re.sub(r'[^a-z0-9]+',' ',str(s).casefold()).split())


def slug(s):
    return re.sub(r'[^A-Za-z0-9_\-]+','_',s.strip())[:100].strip('_') or 'unlabeled'


def load_csv(path):
    if not path.is_file(): return []
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def put_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader();w.writerows(rows)


def target_rows():
    t=load_csv(CLASS_CSV)
    assert len(t)==97 and len({x['crop'] for x in t})==18
    return t


def setup(t):
    OUT.mkdir(parents=True,exist_ok=True)
    for x in t:
        for level in ('CO_NHAN_THEO_NGUON','UNG_VIEN_CHUA_XAC_MINH'):
            (OUT / level / x['crop'] / x['class']).mkdir(parents=True,exist_ok=True)


def request(url,timeout=16,max_bytes=8*1024*1024):
    req=urllib.request.Request(url,headers=HTTP_HEADERS)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        status=getattr(r,'status',200)
        if status!=200:raise RuntimeError(f'HTTP {status}')
        length=int(r.headers.get('Content-Length','0') or '0')
        if length>max_bytes: raise ValueError(f'Qua lon {length} bytes')
        data=r.read(max_bytes+1)
        if len(data)>max_bytes: raise ValueError('Vuot gioi han tai')
        return data


def json_request(url,timeout=15):
    return json.loads(request(url,timeout=timeout,max_bytes=3*1024*1024).decode('utf-8'))


def stream_download(url,dest,max_mb=1400):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and zipfile.is_zipfile(dest):return dest
    temp=dest.with_suffix('.part')
    try:
        req=urllib.request.Request(url,headers=HTTP_HEADERS)
        with urllib.request.urlopen(req,timeout=55) as resp,temp.open('wb') as file:
            size=int(resp.headers.get('Content-Length','0') or '0')
            if size>max_mb*1024*1024: raise ValueError(f'Archive >{max_mb}MB')
            total=0
            while True:
                part=resp.read(256*1024)
                if not part:break
                total+=len(part)
                if total>max_mb*1024*1024:raise ValueError('Archive qua lon')
                file.write(part)
        if not zipfile.is_zipfile(temp): raise ValueError('Tep tai khong phai ZIP')
        temp.replace(dest)
    finally:
        temp.unlink(missing_ok=True)
    return dest


def check_image(raw,minimum=320):
    if len(raw)<6000 or len(raw)>14*1024*1024:return None
    try:
        from PIL import Image, ImageStat
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            w,h=im.size
            if min(w,h)<minimum or max(w,h)>15000:return None
            if im.mode == 'RGBA':
                alpha=im.getchannel('A')
                if alpha.getextrema()[0]<220:return None
            # Loai anh co goc gan trang dong nhat (anh scan tren nen trang)
            rgb=im.convert('RGB'); smaller=rgb.resize((32,32))
            corners=[smaller.getpixel((x,y)) for x,y in ((0,0),(31,0),(0,31),(31,31))]
            if sum(min(pixel)>241 for pixel in corners)>=3:return None
            if sum(max(pixel)<12 for pixel in corners)>=3:return None
            return w,h
    except ImportError:
        # Khong co PIL => khong the kiem tra do net, bo qua anh tranh chan doan sai
        raise RuntimeError('Can cai Pillow: python -m pip install Pillow')
    except Exception:return None


def save_photo(raw,cls,bucket,src,src_label,src_url,license,rows,hashes,notes='',format_hint='.jpg'):
    meta=check_image(raw)
    if meta is None: return False
    sha=hashlib.sha256(raw).hexdigest()
    if sha in hashes:return False
    suffix=format_hint.lower()
    if suffix not in EXTS: suffix='.jpg'
    if suffix=='.webp': suffix='.webp'
    record=CLASSES[cls]
    folder=OUT/bucket/record['crop']/cls
    folder.mkdir(parents=True,exist_ok=True)
    filename=f'{slug(src)[:22]}_{sha[:16]}{suffix}'
    path=folder/filename
    path.write_bytes(raw)
    rows.append(dict(**{'class':cls,'crop':record['crop'],'condition':record['condition'],
        'bucket':bucket,'file':str(path.relative_to(OUT)).replace('\\','/'),
        'source':src,'source_label':src_label,'source_url':src_url,'license':license,
        'width':meta[0],'height':meta[1], 'sha256':sha,
        'status':'CAN_DUYET_NHAN_VA_PHONG_NEN','note':notes}))
    hashes.add(sha)
    return True


def tally(rows):
    d=Counter()
    for r in rows:
        path=OUT/r['file']
        if path.exists():d[(r['class'],r['bucket'])]+=1
    return d


def curated(rows,hashes,events,target):
    print('\n[1] Tai anh nguon da duoc bien tap...')
    counts=tally(rows)
    for r in load_csv(LEGACY):
        label=r.get('Nhan','')
        crop=r.get('Cay','')
        cls=next((x for x in CLASSES if CLASSES[x]['crop']==crop and
                   norm(CLASSES[x]['condition'])==norm(label)),None)
        if cls is None:continue
        if counts[(cls,'CO_NHAN_THEO_NGUON')]>=target:continue
        if r.get('Trang_thai_nhan','')!='Theo nguon':continue
        url=r['Anh_URL']
        try:
            raw=request(url,timeout=14,max_bytes=14*1024*1024)
            suffix=PurePosixPath(urllib.parse.urlsplit(url).path).suffix.lower()
            if save_photo(raw,cls,'CO_NHAN_THEO_NGUON','WEB_NGUON',label,r['Nguon_URL'],
                          'Xem dieu kien website goc',rows,hashes,'Ghi nhan tu nguon, chua tham dinh doc lap',suffix):
                counts[(cls,'CO_NHAN_THEO_NGUON')]+=1
                print(' +',cls)
        except Exception as e:
            events.append(dict(source='WEB_NGUON',class_name=cls,url=url,error=str(e)[:180]))
    return counts


def plantdoc_class(path):
    p=PurePosixPath(path.replace('\\','/'))
    parts=[norm(t) for t in p.parts]
    for i, part in enumerate(parts):
        if part in ('train','test') and i+1<len(parts):
            return PLANTDOC.get(parts[i+1]),p.parts[i+1]
    return None,None


def import_zip(archive,source,link,license,rows,hashes,events,target):
    counts=tally(rows)
    source_groups=defaultdict(list)
    try:
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir() or info.file_size>14*1024*1024:continue
                ext=PurePosixPath(info.filename).suffix.lower()
                if ext not in EXTS:continue
                if source=='PLANTDOC':
                    cls,orig=plantdoc_class(info.filename)
                else:
                    cls,orig=field_class(info.filename,source)
                if cls and cls in CLASSES:
                    source_groups[cls].append((info,orig,ext))
            rng=random.Random(7092026)
            for cls in sorted(source_groups):
                if counts[(cls,'CO_NHAN_THEO_NGUON')]>=target:continue
                items=source_groups[cls]
                rng.shuffle(items)
                for info,orig,ext in items:
                    if counts[(cls,'CO_NHAN_THEO_NGUON')]>=target:break
                    try:
                        raw=z.read(info)
                        if save_photo(raw,cls,'CO_NHAN_THEO_NGUON',source,orig,link,
                                      license,rows,hashes,'Nhan tu thu muc du lieu nguon, can doi chieu',ext):
                            counts[(cls,'CO_NHAN_THEO_NGUON')]+=1
                    except Exception as e:
                        events.append(dict(source=source,class_name=cls,url=info.filename,error=str(e)[:180]))
                print(f' {cls}: {counts[(cls,"CO_NHAN_THEO_NGUON")]}/{target} anh co nhan nguon')
    except Exception as e:
        events.append(dict(source=source,class_name='',url=str(archive),error=str(e)[:180]))
    return counts


def plantdoc(rows,hashes,events,target):
    print('\n[2] Tai tap anh thuc dia PlantDoc...')
    archive=CACHE/'PlantDoc.zip'
    link='https://github.com/pratikkayal/PlantDoc-Dataset'
    urls=['https://codeload.github.com/pratikkayal/PlantDoc-Dataset/zip/refs/heads/master',
          'https://github.com/pratikkayal/PlantDoc-Dataset/archive/refs/heads/master.zip']
    try:
        if not archive.exists():
            error=None
            for url in urls:
                try:
                    stream_download(url,archive,max_mb=650)
                    error=None;break
                except Exception as e:
                    error=e
            if error:raise error
        import_zip(archive,'PLANTDOC',link,'CC BY 4.0',rows,hashes,events,target)
    except Exception as e:
        events.append(dict(source='PLANTDOC',class_name='',url=link,error=str(e)[:180]))
        print(' Loi tai PlantDoc:',str(e)[:170])


def field_class(path,source):
    p=PurePosixPath(path.replace('\\','/'))
    seg=[norm(s) for s in p.parts[:-1]]
    if not seg:return None,None
    if any('augment' in x or 'processed' in x or 'resized' in x or
           'synthetic' in x or 'overlay' in x or 'annotation' in x or
           'crop only' in x for x in seg):
        return None,None
    if source=='Durian_VinhLong_RealField':
        return None,None  # Anh la, than, qua tron chung; nhan benh khac 97 lop
    for crop,alias in FIELD_MAP.items():
        if not alias:continue
        detected=any(x==norm(crop) or x.startswith(norm(crop)+' ') for x in seg)
        if not detected and SOURCE_CROP.get(source)!=crop:continue
        for text in reversed(seg):
            field=text
            if field.startswith(norm(crop)+' '):
                field=field[len(norm(crop))+1:]
            # Nhan nguon phai trung CHINH XAC. Vi du 'black sigatoka' khac
            # 'sigatoka', 'bacterial spot' khac 'bacterial speck'.
            if field in alias and alias[field] is not None:
                return alias[field],text
        return None,None
    return None,None


def mendeley_dataset_files(id,version):
    # API công khai của Mendeley thay đổi theo endpoint; thử hai dạng.
    endpoints=[f'https://api.data.mendeley.com/datasets/{id}/files?version={version}',
               f'https://api.data.mendeley.com/datasets/publics/{id}?version={version}',
               f'https://api.data.mendeley.com/datasets/{id}?version={version}']
    last=None
    for ep in endpoints:
        try:
            data=json_request(ep,timeout=10)
            if isinstance(data,list):return data
            if isinstance(data,dict):
                for key in ('files','items','results','data'):
                    if isinstance(data.get(key),list):return data[key]
            last='Khong co danh sach tep'
        except Exception as e:last=e
    raise RuntimeError('Mendeley API khong co danh sach file: '+str(last)[:140])


def mendeley(rows,hashes,events,target):
    print('\n[3] Lay bo thuc dia Mendeley bo sung cho nhieu cay...')
    consecutive_errors=0
    for dataset_id,version,name,link in MENDELEY_SOURCES:
        if name=='Durian_VinhLong_RealField':
            print(' Bo qua nhan benh Sau rieng vi bo du lieu nay khong khop 97 lop va chua anh than/qua.')
            continue
        files=[]
        try:
            files=mendeley_dataset_files(dataset_id,version)
            consecutive_errors=0
        except Exception as e:
            consecutive_errors+=1
            events.append(dict(source=name,class_name='',url=link,error=str(e)[:180]))
            print(' API Mendeley chua hoat dong:',name)
            if consecutive_errors>=3:
                print(' Khong truy cap duoc API cho 3 nguon lien tiep; bo qua cac nguon con lai.')
                break
        local_dir=CACHE/name
        local_dir.mkdir(parents=True,exist_ok=True)
        archives=list(local_dir.glob('*.zip'))
        if not archives:
            for entry in files:
                filename=entry.get('filename') or entry.get('name') or ''
                if not filename.lower().endswith('.zip'): continue
                if 'augment' in filename.lower():continue
                file_id=entry.get('id') or entry.get('file_id')
                details=entry.get('content_details') or {}
                candidates=[details.get('download_url'),entry.get('download_url')]
                if file_id:
                    candidates += [f'https://api.data.mendeley.com/datasets/{dataset_id}/files/{file_id}/file_downloaded?version={version}',
                                   f'https://api.data.mendeley.com/datasets/{dataset_id}/files/{file_id}/download?version={version}']
                for url in [x for x in candidates if x]:
                    try:
                        path=stream_download(url,local_dir/slug(filename).replace('_zip','.zip'),max_mb=1400)
                        archives.append(path)
                        break
                    except Exception as e:
                        events.append(dict(source=name,class_name='',url=url,error=str(e)[:150]))
                if archives:break
        for file in archives:
            import_zip(file,name,link,'CC BY 4.0 (kiem tra trang nguon)',rows,hashes,events,target)
        if not archives:
            print(' Chua lay duoc ZIP',name,'– co the tai thu cong tai:',link)


def commons_search(query):
    params={'action':'query','format':'json','formatversion':2,
        'generator':'search','gsrsearch':query,'gsrnamespace':6,
        'gsrlimit':30,'prop':'imageinfo|categories','cllimit':20,
        'iiprop':'url|size|mime|extmetadata','iiurlwidth':1400}
    url='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params)
    result=json_request(url,timeout=15)
    return result.get('query',{}).get('pages',[]) or []


def disease_key(record):
    return norm(record['condition'])


def text_matches(record,text):
    text=norm(text)
    crop=record['crop']
    if not any(s in text for s in CROP_SYNONYMS[crop]):return False
    cond=disease_key(record)
    if cond=='healthy':
        # Anh la canh cay khong phai chung nhan la khoe, chi la ung vien.
        return 'leaf' in text or 'leaves' in text
    parts=[x for x in cond.split() if x not in ('leaf','leaves','two','spotted','the','and')]
    if not parts:return False
    if cond=='die back':parts=['dieback']
    if cond=='brown rust':parts=['rust']
    if cond=='red rust':parts=['rust']
    return all(x in text for x in parts)


def commons(rows,hashes,events,target):
    print('\n[4] Tim anh tren Wikimedia Commons cho 97 nhan (chi la ung vien)...')
    counts=tally(rows)
    errors=0
    for i,(cls,r) in enumerate(CLASSES.items(),1):
        total=counts[(cls,'CO_NHAN_THEO_NGUON')]+counts[(cls,'UNG_VIEN_CHUA_XAC_MINH')]
        need=target-total
        if need<=0:
            print(f' [{i:2}/97] {cls}: da co {target} anh, bo qua');continue
        crop=r['crop'].replace('__',' ')
        cond=r['condition_display']
        query=f'{crop} {cond} leaf'
        try:
            pages=commons_search(query)
            errors=0
        except Exception as e:
            errors+=1
            events.append(dict(source='WIKIMEDIA',class_name=cls,url=query,error=str(e)[:170]))
            print(f' [{i:2}/97] API loi {cls}: {str(e)[:80]}')
            # Nếu máy không có mạng, dừng ngay thay vì đợi 97*15 giây.
            if errors>=3:
                print(' Khong truy cap duoc Wikimedia, ket thuc buoc nay.');break
            continue
        random.Random(i*2026).shuffle(pages)
        accepted=0
        for page in pages:
            if accepted>=need:break
            info=page.get('imageinfo') or []
            if not info:continue
            ii=info[0]
            mime=ii.get('mime','')
            if mime not in ('image/jpeg','image/png','image/webp'):continue
            if min(int(ii.get('width',0)),int(ii.get('height',0)))<320:continue
            meta=ii.get('extmetadata') or {}
            lic=((meta.get('LicenseShortName') or {}).get('value') or '').strip()
            desc=((meta.get('ImageDescription') or {}).get('value') or '')
            category=' '.join(c.get('title','') for c in page.get('categories') or [])
            desc=re.sub('<[^>]+>',' ',html.unescape(desc))
            text=' '.join([page.get('title',''),desc,category])
            if not text_matches(r,text):continue
            image_url=ii.get('thumburl') or ii.get('url')
            if not image_url:continue
            # Respect licenses; skip unknown/no-license and unsuitable licenses.
            if not lic or any(z in lic.casefold() for z in ('all rights','copyrighted','fair use')):continue
            try:
                raw=request(image_url,timeout=14,max_bytes=14*1024*1024)
                suffix=PurePosixPath(urllib.parse.urlsplit(image_url).path).suffix.lower()
                file_link='https://commons.wikimedia.org/wiki/'+urllib.parse.quote(page.get('title','').replace(' ','_'))
                if save_photo(raw,cls,'UNG_VIEN_CHUA_XAC_MINH','WIKIMEDIA',page.get('title',''),file_link,lic,
                              rows,hashes,'Tim theo tu khoa; chua chung minh anh dung benh/ngoai vuon',suffix):
                    accepted+=1
            except Exception as e:
                events.append(dict(source='WIKIMEDIA',class_name=cls,url=image_url,error=str(e)[:120]))
        print(f' [{i:2}/97] {cls}: +{accepted} ung vien (can {need})')
        time.sleep(0.15)


def report(rows,events,targets,n):
    by=tally(rows)
    result=[]
    for r in targets:
        cls=r['class']
        official=by[(cls,'CO_NHAN_THEO_NGUON')]
        candidate=by[(cls,'UNG_VIEN_CHUA_XAC_MINH')]
        short=max(0,n-official)
        result.append({'loai_cay':r['crop_vi'],'ma_cay':r['crop'],'nhan_ai':cls,
                'so_anh_can':n,'anh_nhan_theo_nguon':official,'anh_ung_vien_chua_xac_minh':candidate,
                'con_thieu_anh_co_nhan':short,'ket_qua':'DU_NHAN_NGUON_CAN_KIEM_TRA_TRUNG' if short==0 else 'THIEU_ANH_CO_NHAN',
                'duong_dan':str(OUT/'CO_NHAN_THEO_NGUON'/r['crop']/cls),
                'tim_anh':r['google_images']})
    put_csv(OUT/'BAO_CAO_97_LOP.csv',result,list(result[0]))
    crop_summaries=[]
    for crop in sorted(set(r['crop'] for r in targets)):
        bunch=[q for q in result if q['ma_cay']==crop]
        crop_summaries.append({'loai_cay':bunch[0]['loai_cay'],'ma_cay':crop,
          'so_nhan':len(bunch),'nhan_du_5_anh_co_nhan':sum(x['con_thieu_anh_co_nhan']==0 for x in bunch),
          'anh_co_nhan_nguon':sum(x['anh_nhan_theo_nguon'] for x in bunch),
          'ung_vien_can_xac_minh':sum(x['anh_ung_vien_chua_xac_minh'] for x in bunch),
          'anh_co_nhan_con_thieu':sum(x['con_thieu_anh_co_nhan'] for x in bunch)})
    put_csv(OUT/'BAO_CAO_18_CAY.csv',crop_summaries,list(crop_summaries[0]))
    put_csv(EVENTS,events,['source','class_name','url','error'])
    create_gallery(result,rows)
    bycroplist=len(set(x['ma_cay'] for x in result))
    print('\n'+'='*64)
    print('DANH SACH CO:',bycroplist,'loai cay,',len(result),'nhan, muc tieu',len(result)*n,'anh CO NHAN')
    print('Co nhan theo nguon:',sum(q['anh_nhan_theo_nguon'] for q in result))
    print('Anh ung vien chua kiem:',sum(q['anh_ung_vien_chua_xac_minh'] for q in result))
    print('Nhom DU 5 ANH CO NHAN:',sum(q['con_thieu_anh_co_nhan']==0 for q in result),'/',len(result))
    print('Con thieu:',sum(q['con_thieu_anh_co_nhan'] for q in result),'anh co nhan')
    print('Mo:',OUT/'BAO_CAO_97_LOP.csv')
    print('Xem hinh:',OUT/'THU_VIEN_ANH.html')
    print('!! Khong co anh tuong duong voi khang dinh nhan benh dung 100%')


def create_gallery(summary,rows):
    grouped=defaultdict(list)
    for r in rows:
        if (OUT/r['file']).exists(): grouped[r['class']].append(r)
    css='''body{font:14px/1.5 system-ui,Arial,sans-serif;background:#f5f7fb;color:#222;max-width:1150px;margin:20px auto;padding:15px}header,section{background:white;padding:15px 19px;margin:12px 0;border-radius:12px;border:1px solid #e1e5ed}h1{font-size:24px}h2{font-size:17px} .photo{display:inline-block;width:180px;vertical-align:top;margin:5px;padding:5px;background:#f7f8fa;border-radius:9px;overflow-wrap:anywhere}img{width:170px;height:130px;object-fit:cover;border-radius:6px}small{font-size:11px;color:#666}.note{background:#fff0cf;padding:12px;border-radius:8px}.bad{color:#a02b24}.good{color:#096b35}a{color:#1257a5}'''
    page=['<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>18 cây 97 lớp — ảnh ngoài train</title><style>'+css+'</style><body>',
      '<header><h1>Thư viện ảnh ngoài Train: 18 cây / 97 nhãn</h1><p class="note">Chỉ dùng để tuyển dữ liệu. Ảnh theo nhãn nguồn không có nghĩa đã xác minh không trùng Train. Wikimedia là ứng viên chưa kiểm tra đúng bệnh, lá, phông nền. Không dùng ứng viên để tính Accuracy.</p></header>']
    for r in summary:
        cls=r['nhan_ai']
        page.append('<section><h2>'+html.escape(cls)+' — <span class="'+('good' if not r['con_thieu_anh_co_nhan'] else 'bad')+'">'+str(r['anh_nhan_theo_nguon'])+'/'+str(r['so_anh_can'])+' ảnh theo nhãn nguồn</span></h2>')
        for item in grouped.get(cls,[]):
            rel=item['file']
            page.append('<div class="photo"><a href="'+html.escape(rel,quote=True)+'"><img loading="lazy" src="'+html.escape(rel,quote=True)+'"></a><small>'+html.escape(item['bucket'])+'<br><a href="'+html.escape(item['source_url'],quote=True)+'">Nguồn</a> · '+html.escape(item['license'])+'</small></div>')
        if not grouped.get(cls):page.append('<p>Chưa có ảnh phù hợp.</p>')
        page.append('<p><a href="'+html.escape(r['tim_anh'],quote=True)+'">Xem ảnh tìm kiếm bổ sung</a></p></section>')
    page.append('</body></html>')
    (OUT/'THU_VIEN_ANH.html').write_text('\n'.join(page),encoding='utf-8')


def duplicate_check(rows,events,targets,n):
    origin=ROOT/'dataset_review'/'dataset_97_lop'/'split_97_lop'/'dataset_manifest_SPLIT_DU_KIEN.csv'
    if not origin.exists():
        print('KHONG TIM THAY MANIFEST ORIGINAL:',origin)
        return
    print('Dang quet ma bam goc, chi check SHA256 exact; co the can nhieu phut...')
    external=defaultdict(list)
    for r in rows:
        path=OUT/r['file']
        if path.exists():external[r['sha256']].append(r)
    matches=[];scanned=0;missing=0
    with origin.open(encoding='utf-8-sig',newline='') as f:
        for orig in csv.DictReader(f):
            src=Path(orig['path'])
            if not src.is_file():missing+=1;continue
            try:
                h=hashlib.sha256()
                with src.open('rb') as fp:
                    for chunk in iter(lambda:fp.read(1024*1024),b''):h.update(chunk)
                key=h.hexdigest();scanned+=1
                if key in external:
                    for r in external[key]:
                        matches.append(dict(anh_ngoai=r['file'],nguon=r['source'],
                          nhan_ngoai=r['class'],anh_train=str(src),
                          nhan_train=orig['class'],split=orig['split'],sha256=key))
                if scanned%10000==0:print('  Da quet',scanned,'tep')
            except Exception as e:missing+=1
    fields=['anh_ngoai','nguon','nhan_ngoai','anh_train','nhan_train','split','sha256']
    put_csv(OUT/'TRUNG_VOI_TRAIN_VAL_TEST.csv',matches,fields)
    print('Da quet:',scanned,'Khong mo duoc:',missing,'Trung SHA256:',len(matches))
    print('CHU Y: khong trung byte van co the trung nhung resize/crop/augment.')


def main():
    pa=argparse.ArgumentParser()
    group=pa.add_mutually_exclusive_group(required=True)
    group.add_argument('--fast',action='store_true',help='Nguon da chon + PlantDoc + Wikimedia 97 nhan')
    group.add_argument('--mendeley',action='store_true',help='Tai them bo thuc dia goc Mendeley')
    group.add_argument('--all',action='store_true',help='Thuc hien ca fast va Mendeley')
    group.add_argument('--report',action='store_true',help='Thong ke lai sau khi chay')
    group.add_argument('--check-duplicates',action='store_true',help='So SHA256 voi Train/Val/Test')
    pa.add_argument('--target',type=int,default=5)
    pa.add_argument('--output',default='')
    args=pa.parse_args()
    global ROOT,OUT,CACHE,INDEX,EVENTS,CLASSES
    if args.output:
        OUT=Path(args.output);CACHE=OUT/'_cache';INDEX=OUT/'DANH_SACH_ANH.csv';EVENTS=OUT/'LOI_VA_SU_KIEN.csv'
    targets=target_rows()
    CLASSES={r['class']:r for r in targets}
    setup(targets)
    rows=load_csv(INDEX)
    rows=[r for r in rows if r.get('class') in CLASSES and (OUT/r.get('file','')).is_file()]
    hashes={r['sha256'] for r in rows if r.get('sha256')}
    events=load_csv(EVENTS)
    if args.fast or args.all:
        curated(rows,hashes,events,args.target)
        plantdoc(rows,hashes,events,args.target)
        commons(rows,hashes,events,args.target)
    if args.mendeley or args.all:
        mendeley(rows,hashes,events,args.target)
    put_csv(INDEX,rows,FIELDS)
    if args.check_duplicates:duplicate_check(rows,events,targets,args.target)
    report(rows,events,targets,args.target)

if __name__=='__main__':main()
