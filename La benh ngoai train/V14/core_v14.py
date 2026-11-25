# -*- coding: utf-8 -*-
"""Read-only crop/disease inference, independent of the old Streamlit app."""
from __future__ import annotations
import io
from pathlib import Path
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from model_architecture_v14 import get_mobilenet_v2_model

PROJECT = Path(r'D:\DoanHMvUD\La benh ngoai train')
CROP_MODEL = Path(r'D:\DoanHMvUD\training_runs\field18_v3_safe\field18_best_candidate_v3.pth')
DISEASE_MODEL = Path(r'D:\DoanHMvUD\training_runs\field97_v2_safe\field97_best_candidate.pth')
NORMALIZE = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((.485,.456,.406), (.229,.224,.225)),
])

def crop_from_name(label: str) -> str:
    if label.startswith('Tomato_'): return 'Tomato'
    if label.startswith('Pepper__bell___'): return 'Pepper__bell'
    return label.split('___', 1)[0]

def read_metadata(file: Path, required_count: int) -> list[str]:
    checkpoint = torch.load(file, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint phai la dict.')
    for k in ('model_state_dict', 'class_names'):
        if k not in checkpoint: raise ValueError(f'Checkpoint thieu {k}: {file}')
    labels = list(checkpoint['class_names'])
    if len(labels) != required_count or len(set(labels)) != required_count:
        raise ValueError(f'Checkpoint khong co du {required_count} nhan: {file}')
    return labels

def validate_checkpoint_paths(crop_path=CROP_MODEL, disease_path=DISEASE_MODEL):
    p1, p2 = Path(crop_path), Path(disease_path)
    for p in (p1,p2):
        if not p.is_file(): raise FileNotFoundError(f'Khong tim thay checkpoint: {p}')
    names18 = read_metadata(p1,18)
    names97 = read_metadata(p2,97)
    disease_crops = {crop_from_name(n) for n in names97}
    if set(names18) != disease_crops:
        raise ValueError('Danh sach 18 cay trong checkpoint V3 va V2 khong khop.')
    return names18, names97

def load_model(file, num_classes, device):
    checkpoint = torch.load(file, map_location='cpu', weights_only=True)
    labels = list(checkpoint['class_names'])
    m = get_mobilenet_v2_model(num_classes, pretrained=False)
    m.load_state_dict(checkpoint['model_state_dict'], strict=True)
    m.to(device).eval()
    return m, labels

def get_models(device):
    validate_checkpoint_paths()
    cm, names18 = load_model(CROP_MODEL,18,device)
    dm, names97 = load_model(DISEASE_MODEL,97,device)
    return cm, names18, dm, names97

def prepare_image_bytes(content: bytes):
    with Image.open(io.BytesIO(content)) as raw:
        im = ImageOps.exif_transpose(raw).convert('RGB')
    if min(im.size)<100:
        raise ValueError('Anh qua nho (can moi chieu >=100 px).')
    if im.width*im.height>30_000_000:
        im.thumbnail((4096,4096))
    tensor = NORMALIZE(im).unsqueeze(0)
    return im, tensor

def rank_crops(logits, names, k=3):
    scores = torch.softmax(logits.detach().cpu(),dim=-1)
    ids = torch.argsort(scores,descending=True)[:k].tolist()
    return [{'crop':names[i], 'score':round(float(scores[i]),4)} for i in ids]

def rank_diseases(logits, labels, crop, k=3):
    ids = [i for i,n in enumerate(labels) if crop_from_name(n) == crop]
    if not ids:raise ValueError('Khong co nhan benh cho cay: '+crop)
    # This is a *conditional, uncalibrated* relative score; never present as diagnostic confidence.
    scores = torch.softmax(logits.detach().cpu()[ids],dim=-1)
    idxs = torch.argsort(scores,descending=True)[:k].tolist()
    return [{'label': labels[ids[j]], 'relative_score':round(float(scores[j]),4)} for j in idxs]

CROP_VI = {
    'Tomato':'Cà chua','Coffee':'Cà phê','Rubber':'Cao su','Lemon':'Chanh',
    'Tea':'Chè','Banana':'Chuối','Cashew':'Điều','Papaya':'Đu đủ','Potato':'Khoai tây',
    'Rice':'Lúa nước','Sugarcane':'Mía','Jackfruit':'Mít','Corn':'Ngô','Guava':'Ổi',
    'Chili':'Ớt cay','Pepper__bell':'Ớt chuông','Durian':'Sầu riêng','Mango':'Xoài'
}
def crop_label(code): return f'{CROP_VI.get(code,code)} ({code})'
