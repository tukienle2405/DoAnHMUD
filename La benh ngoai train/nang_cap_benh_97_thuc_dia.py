# -*- coding: utf-8 -*-
"""Fine-tune 97 classes on LOCKED field TRAIN with original-data replay.

No leakage: field TEST is never used for training, selection, thresholds or early stopping.
No automatic Streamlit integration. Original checkpoints are never overwritten.

Runs on Windows with project root D:\\DoanHMvUD and Python 3.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

FIELD_LABELS = {
    'Bell_pepper leaf': 'Pepper__bell___healthy',
    # Bell_pepper leaf spot excluded: no verified disease label
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
    # Tomato leaf yellow virus excluded: disease mapping not verified
    'Tomato mold leaf': 'Tomato_Leaf_Mold',
    'Tomato two spotted spider mites leaf': 'Tomato_Spider_mites_Two_spotted_spider_mite',
}
UNVERIFIED_GROUPS = {'Bell_pepper leaf spot', 'Tomato leaf yellow virus'}
CROPS = set('Banana Cashew Chili Coffee Corn Durian Guava Jackfruit Lemon Mango Papaya Pepper__bell Potato Rice Rubber Sugarcane Tea Tomato'.split())
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}


def crop_of(label):
    if label.startswith('Pepper__bell___'):
        return 'Pepper__bell'
    if label.startswith('Tomato_'):
        return 'Tomato'
    return label.split('___', 1)[0]


def csv_rows(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def pct(correct, total):
    return round(correct * 100.0 / total, 2) if total else None


def find_root():
    # The code file may be inside 'La benh ngoai train', one level below project root.
    source = Path(__file__).resolve()
    for p in (source.parent, *source.parents):
        if (p / 'training_runs').is_dir():
            return p
    return Path(r'D:\DoanHMvUD')


def find_field_csv(root, explicit):
    opts = [Path(explicit)] if explicit else [
        root / 'external_field_pool_v2' / 'EXTERNAL_SPLIT_LOCKED.csv',
        Path(__file__).parent / 'EXTERNAL_SPLIT_LOCKED.csv',
    ]
    for p in opts:
        if p.is_file():
            return p
    raise FileNotFoundError('Khong thay EXTERNAL_SPLIT_LOCKED.csv. Dung --field_csv PATH')


def find_manifest(root, explicit):
    opts = [Path(explicit)] if explicit else [
        root / 'dataset_review' / 'dataset_97_lop' / 'split_97_lop' / 'dataset_manifest_SPLIT_DU_KIEN.csv',
        Path(__file__).parent / 'dataset_manifest_SPLIT_DU_KIEN(1).csv',
    ]
    for p in opts:
        if p.is_file():
            return p
    raise FileNotFoundError('Khong thay manifest 97 lop. Dung --manifest PATH')


def split_locked(raw):
    needed = {'path', 'group', 'crop', 'split', 'reason', 'cluster'}
    if not raw or not needed.issubset(raw[0]):
        raise ValueError(f'EXTERNAL_SPLIT_LOCKED.csv phai co cot {sorted(needed)}')
    paths = [r['path'].casefold() for r in raw]
    if len(set(paths)) != len(paths):
        raise ValueError('Duong dan anh bi trung trong locked CSV')
    splits = defaultdict(list)
    clusters = defaultdict(set)
    for r in raw:
        if r['split'] not in {'train', 'val', 'test', 'exclude'}:
            raise ValueError(f'Invalid split {r["split"]}')
        if r['reason'] != 'OK':
            if r['split'] != 'exclude':
                raise ValueError('Anh co reason != OK khong duoc o train/val/test')
            continue
        if r['group'] not in FIELD_LABELS and r['group'] not in UNVERIFIED_GROUPS:
            raise ValueError('Nhom anh chua duoc anh xa nhan: ' + r['group'])
        expected_crop = crop_of(FIELD_LABELS[r['group']]) if r['group'] in FIELD_LABELS else (
            'Tomato' if r['group'] == 'Tomato leaf yellow virus' else 'Pepper__bell')
        if r['crop'] != expected_crop:
            raise ValueError('Nhom va cay khong khop: ' + r['path'])
        if r.get('cluster'):
            clusters[r['cluster']].add(r['split'])
        splits[r['split']].append(r)
    leaking = [x for x, s in clusters.items() if len(s) != 1]
    if leaking:
        raise ValueError(f'RO RI: {len(leaking)} cum anh xuat hien o nhieu split!')
    return splits


def make_field_samples(raw):
    """Only classes verified by previous source labels may train/evaluate disease."""
    out = []
    for r in raw:
        if r['group'] in FIELD_LABELS:
            out.append({'path': r['path'], 'label': FIELD_LABELS[r['group']],
                        'group': r['group'], 'crop': r['crop'], 'cluster': r.get('cluster', ''),
                        'source_domain': 'field'})
    return out


def check_test_csv_if_exists(root, field_splits, explicit=None):
    candidates = [Path(explicit)] if explicit else [
        Path(__file__).parent / 'DU_DOAN_TUNG_ANH(1).csv',
        Path(__file__).parent / 'DU_DOAN_TUNG_ANH.csv',
    ]
    for p in candidates:
        if p.is_file():
            external = csv_rows(p)
            locked = {r['path'].casefold() for r in field_splits['test']}
            seen = {r['path'].casefold() for r in external}
            if seen != locked:
                raise ValueError('Test CSV khong trung test locked. KHONG DANH GIA!')
            for row in external:
                if row['benh_theo_nhan_nguon'] not in {'CHUA_XAC_MINH', ''}:
                    expected = FIELD_LABELS.get(row['nhom_plantdoc'])
                    if expected != row['benh_theo_nhan_nguon']:
                        raise ValueError('Nhan test khong trung nhan nhom da xac minh: ' + row['path'])
            return p
    return None


def scan_old_manifest(path, class_names=None):
    if not path.is_file():
        raise FileNotFoundError(path)
    allowed = set(class_names) if class_names else None
    out = defaultdict(lambda: defaultdict(list))
    stats = Counter()
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sp = row.get('split', '')
            if sp not in {'train', 'val'}:
                continue
            if row.get('needs_label_review', 'NO').upper() == 'YES':
                stats['skip_needs_label_review'] += 1
                continue
            cl = row['class']
            if allowed is not None and cl not in allowed:
                raise ValueError('Manifest co class khong ton tai trong checkpoint: ' + cl)
            out[sp][cl].append({'path': row['path'], 'label': cl, 'source_domain': 'old'})
            stats['included_' + sp] += 1
    return out, stats


def inspect_unreviewed_class_coverage(path):
    by_class = defaultdict(lambda: Counter())
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r.get('split') in {'train', 'val', 'test'}:
                by_class[r['class']][r.get('needs_label_review', 'NO').upper()] += 1
    risky = []
    for name, c in sorted(by_class.items()):
        if c['NO'] == 0:
            risky.append({'class': name, 'needs_label_review': c['YES'], 'verified_images': 0})
    return risky


def check_existing_files(samples, scope):
    missing = [r['path'] for r in samples if not Path(r['path']).is_file()]
    if missing:
        raise FileNotFoundError(f'{scope}: thieu {len(missing)}/{len(samples)} anh. Mau: {missing[0]}')


def audit(field_rows, original_summary, out):
    splits = split_locked(field_rows)
    counts = []
    by_class = defaultdict(lambda: Counter())
    for split in ('train', 'val', 'test'):
        for r in splits[split]:
            if r['group'] in FIELD_LABELS:
                by_class[FIELD_LABELS[r['group']]][split] += 1
    for c in sorted(FIELD_LABELS.values()):
        counts.append({'class': c, 'train': by_class[c]['train'],
                       'val': by_class[c]['val'], 'test': by_class[c]['test']})
    write_csv(out / 'PHAN_BO_NHAN_THUC_DIA.csv', counts)
    num = lambda sp: sum(1 for r in splits[sp] if r['group'] in FIELD_LABELS)
    totals = {sp: len(splits[sp]) for sp in ('train', 'val', 'test')}
    known = {sp: num(sp) for sp in ('train', 'val', 'test')}
    overview = {
        'locked_counts': totals,
        'known_disease_counts': known,
        'unverified_disease_counts': {k: totals[k] - known[k] for k in totals},
        'excluded_by_lock': sum(r['split'] == 'exclude' for r in field_rows),
        'field_validated_class_count': len(FIELD_LABELS),
        'field_classes_absent_from_train': [c for c in sorted(by_class) if by_class[c]['train'] == 0],
        'source_test_is_never_used_for_model_selection': True,
        'original_manifest': original_summary,
    }
    save_json(out / 'AUDIT_BO_DU_LIEU.json', overview)
    print('===== KIEM TRA TAP DU LIEU =====')
    print('Lock:', totals, '| Co nhan benh:', known, '| Exclude:', overview['excluded_by_lock'])
    print('Benh khong co anh train thuc dia:', overview['field_classes_absent_from_train'])
    print('Da luu:', out / 'AUDIT_BO_DU_LIEU.json')
    return splits, overview


def load_checkpoint(path, expect, device):
    import torch
    from model_architecture import get_mobilenet_v2_model
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    names = list(ckpt['class_names'])
    if len(names) != expect or len(set(names)) != expect:
        raise ValueError(f'{path} khong co {expect} class_names rieng biet')
    if expect == 18 and set(names) != CROPS:
        raise ValueError('Model 18 khong co du 18 loai cay')
    if expect == 97 and {crop_of(c) for c in names} != CROPS:
        raise ValueError('Model 97 khong co du 18 cay')
    model = get_mobilenet_v2_model(num_classes=expect, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    return model.to(device), names


def make_transform(train=False):
    from torchvision import transforms
    ops = []
    if train:
        ops += [
            transforms.RandomResizedCrop(224, scale=(0.83, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(
                brightness=0.12, contrast=0.12, saturation=0.10)], p=0.5),
        ]
    else:
        ops.append(transforms.Resize((224, 224)))
    ops += [transforms.ToTensor(), transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    return transforms.Compose(ops)


class LeafDataset:
    def __init__(self, samples, indices, train=False):
        self.samples = samples
        self.indices = indices
        self.transform = make_transform(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image, ImageOps
        r = self.samples[idx]
        with Image.open(r['path']) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            im = self.transform(im)
        return im, self.indices[r['label']], int(r.get('source_domain') == 'old')


def build_loader(samples, indices, device, batch_size, train):
    from torch.utils.data import DataLoader
    return DataLoader(LeafDataset(samples, indices, train),
                      batch_size=batch_size, shuffle=train,
                      drop_last=train, num_workers=0,
                      pin_memory=(device.type == 'cuda'))


def choose_field(field_train, rng, count):
    n = Counter(x['label'] for x in field_train)
    weights = [1 / math.sqrt(n[x['label']]) for x in field_train]
    return rng.choices(field_train, weights=weights, k=count)


def choose_old(old_by_class, rng, count):
    classes = sorted(k for k, v in old_by_class.items() if v)
    if len(classes) < 1:
        raise ValueError('Old train khong co lop nhan dang tin cay')
    return [rng.choice(old_by_class[rng.choice(classes)]) for _ in range(count)]


def balanced_val(old_by_class, per_class, seed=2026):
    rng = random.Random(seed)
    return [x for c in sorted(old_by_class)
            for x in rng.sample(old_by_class[c], min(per_class, len(old_by_class[c])))]


def crop_logits_predict(logits, names97, crop):
    import torch
    ids = [i for i, label in enumerate(names97) if crop_of(label) == crop]
    if not ids:
        raise ValueError('Khong co benh cua cay: ' + crop)
    return ids[int(logits[ids].argmax().item())]


def evaluate_field(model97, model18, names97, names18, rows, device, batch_size,
                   save_predictions=False):
    import torch
    model18.eval()
    model97.eval()
    mapper = {c: i for i, c in enumerate(names97)}
    loader = build_loader(rows, mapper, device, batch_size, train=False)
    correct_crop = correct_oracle = correct_joint = correct_free = 0
    total = 0
    output = []
    with torch.inference_mode():
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits97 = model97(x)
            logits18 = model18(x)
            ids18 = logits18.argmax(dim=1).cpu().tolist()
            for i, idx18 in enumerate(ids18):
                ground_idx = int(y[i].item())
                gt_name = names97[ground_idx]
                gt_crop = crop_of(gt_name)
                crop_pred = names18[idx18]
                known_crop_idx = crop_logits_predict(logits97[i], names97, gt_crop)
                pred_idx = crop_logits_predict(logits97[i], names97, crop_pred)
                free_idx = int(logits97[i].argmax().item())
                correct_crop += int(crop_pred == gt_crop)
                correct_oracle += int(known_crop_idx == ground_idx)
                correct_joint += int(pred_idx == ground_idx)
                correct_free += int(free_idx == ground_idx)
                total += 1
                if save_predictions:
                    output.append({'path': rows[total - 1]['path'],
                                   'true_class': gt_name, 'pred_crop': crop_pred,
                                   'pred_class': names97[pred_idx],
                                   'pred_given_true_crop': names97[known_crop_idx],
                                   'correct_crop': int(crop_pred == gt_crop),
                                   'correct_disease_if_correct_crop': int(known_crop_idx == ground_idx),
                                   'correct_joint': int(pred_idx == ground_idx)})
    summary = {'total': total, 'crop_correct': correct_crop,
               'crop_pct': pct(correct_crop, total),
               'joint_correct': correct_joint, 'joint_pct': pct(correct_joint, total),
               'given_true_crop_correct': correct_oracle,
               'given_true_crop_pct': pct(correct_oracle, total),
               'free_97_correct': correct_free, 'free_97_pct': pct(correct_free, total)}
    return summary, output


def evaluate_old(model97, names97, old_val, device, batch_size):
    import torch
    model97.eval()
    mapper = {c: i for i, c in enumerate(names97)}
    loader = build_loader(old_val, mapper, device, batch_size, train=False)
    right = num = 0
    with torch.inference_mode():
        for x, y, _ in loader:
            pred = model97(x.to(device, non_blocking=True)).argmax(dim=1).cpu()
            right += int((pred == y).sum().item())
            num += len(y)
    return {'correct': right, 'total': num, 'pct': pct(right, num)}


def freeze_backbone_bn(model):
    import torch.nn as nn
    for m in model.features.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


def setup_training(model, train_backbone, lr_head, lr_backbone):
    import torch
    for p in model.features.parameters():
        p.requires_grad_(False)
    feature_params = []
    if train_backbone:
        for layer in list(model.features.children())[-4:]:
            for p in layer.parameters():
                p.requires_grad_(True)
                feature_params.append(p)
    for p in model.classifier.parameters():
        p.requires_grad_(True)
    groups = [{'params': list(model.classifier.parameters()), 'lr': lr_head}]
    if feature_params:
        groups.append({'params': feature_params, 'lr': lr_backbone})
    return torch.optim.AdamW(groups, weight_decay=0.01)


def train_epoch(model, teacher, loader, optimizer, device, amp, kd_weight=0.2):
    import torch
    import torch.nn.functional as F
    model.train()
    freeze_backbone_bn(model)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    loss_total = right = total = 0
    for images, labels, is_old in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        is_old = is_old.to(device, non_blocking=True).bool()
        optimizer.zero_grad(set_to_none=True)
        with (torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext()):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            # Teacher distillation only on reliable OLD data to retain 97-way behavior.
            # No teacher constraint on FIELD examples (the domain we need to adapt to).
            if kd_weight > 0 and is_old.any():
                with torch.no_grad():
                    teacher_logits = teacher(images[is_old]).float()
                temperature = 2.0
                kl = F.kl_div(
                    F.log_softmax(logits[is_old].float() / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction='batchmean'
                ) * temperature * temperature
                loss = loss + kd_weight * kl
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_total += float(loss.item()) * len(labels)
        right += int((logits.argmax(1) == labels).sum().item())
        total += len(labels)
    return {'train_loss': round(loss_total / max(total, 1), 5),
            'train_acc_pct': pct(right, total)}


def ensure_no_overlap(splits, manifest_dict):
    field_paths = {r['path'].casefold() for sp in ('train', 'val', 'test') for r in splits[sp]}
    old_paths = {r['path'].casefold() for sp in ('train', 'val')
                 for lst in manifest_dict[sp].values() for r in lst}
    shared = field_paths & old_paths
    if shared:
        raise ValueError('FIELD VÀ OLD dùng trùng đường dẫn: ' + next(iter(shared)))


def summary_by_group(predictions):
    by = defaultdict(list)
    for r in predictions:
        by[r['true_class']].append(r)
    return [{'class': c, 'total': len(vals),
             'correct': sum(r['correct_joint'] for r in vals),
             'joint_pct': pct(sum(r['correct_joint'] for r in vals), len(vals))}
            for c, vals in sorted(by.items())]


def main():
    p = argparse.ArgumentParser(description='97 disease field adaptation with locked split + old replay')
    p.add_argument('--mode', choices=['audit', 'train', 'test'], default='train')
    p.add_argument('--root', default=None, help='Goc du an: D:\\DoanHMvUD')
    p.add_argument('--field_csv', default=None)
    p.add_argument('--manifest', default=None)
    p.add_argument('--model18', default=None)
    p.add_argument('--model97', default=None)
    p.add_argument('--candidate', default=None, help='checkpoint output from previous training for --mode test')
    p.add_argument('--csv_test', default=None, help='Optional DU_DOAN_TUNG_ANH(1).csv for lock cross-check')
    p.add_argument('--out', default=None)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--head_epochs', type=int, default=1)
    p.add_argument('--field_per_epoch', type=int, default=600)
    p.add_argument('--old_per_epoch', type=int, default=600)
    p.add_argument('--old_val_per_class', type=int, default=6)
    p.add_argument('--lr_head', type=float, default=0.0003)
    p.add_argument('--lr_backbone', type=float, default=0.000015)
    p.add_argument('--kd_weight', type=float, default=0.2)
    p.add_argument('--max_old_drop_pct', type=float, default=8.0)
    p.add_argument('--min_joint_gain_pct', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = p.parse_args()

    root = Path(args.root) if args.root else find_root()
    out = Path(args.out) if args.out else root / 'training_runs' / 'field97_v2_safe'
    out.mkdir(parents=True, exist_ok=True)
    field_csv = find_field_csv(root, args.field_csv)
    field_rows = csv_rows(field_csv)
    original_csv = check_test_csv_if_exists(root, split_locked(field_rows), args.csv_test)
    splits, audit_report = audit(field_rows, {'locked_test_csv_consistent': original_csv is not None}, out)
    if args.mode == 'audit':
        manifest_path = find_manifest(root, args.manifest)
        pool, stat = scan_old_manifest(manifest_path)
        risky = inspect_unreviewed_class_coverage(manifest_path)
        save_json(out / 'NHOM_NHAN_97_CAN_XEM_LAI.json', risky)
        print('Original manifest:', dict(stat))
        print('Old train classes:', len(pool['train']))
        print('Risky classes with 0 verified labels:', len(risky), [r['class'] for r in risky])
        print('Original manifest kept OUT of field Test.')
        return

    if args.batch_size < 2:
        raise ValueError('batch_size >= 2 vi model co BatchNorm1d')
    if args.epochs < 1:
        raise ValueError('epochs phai >=1')
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    print('Device:', device)
    ckpt18 = Path(args.model18) if args.model18 else (root / 'training_runs' / 'crop18_field_pilot_20261125_123457' / 'best_field_candidate.pth')
    ckpt97 = Path(args.model97) if args.model97 else (root / 'training_runs' / 'plant97_full_20261118_064241' / 'best_plant_model.pth')
    print('Checkpoint18:', ckpt18)
    print('Checkpoint97:', ckpt97)
    model18, crops18 = load_checkpoint(ckpt18, 18, device)
    model97, classes97 = load_checkpoint(ckpt97, 97, device)
    model18.eval()
    for x in model18.parameters():
        x.requires_grad_(False)
    if not set(FIELD_LABELS.values()).issubset(classes97):
        raise ValueError('Co nhan thuc dia khong ton tai trong 97 checkpoint!')

    field_train = make_field_samples(splits['train'])
    field_val = make_field_samples(splits['val'])
    field_test = make_field_samples(splits['test'])
    check_existing_files(field_val, 'Field validation')
    check_existing_files(field_test, 'Field test')
    mapper = {c: i for i, c in enumerate(classes97)}
    manifest = find_manifest(root, args.manifest)
    old_pool, old_stats = scan_old_manifest(manifest, classes97)
    ensure_no_overlap(splits, old_pool)
    old_val = balanced_val(old_pool['val'], args.old_val_per_class, args.seed)
    if len(old_val) < 1:
        raise ValueError('Original validation khong co nhan tin cay')
    print(f'Old-data replay label verified: {len(old_pool["train"])} / 97 classes; excluded classes need manual review.')
    print('No verified old train:', sorted(set(classes97) - set(old_pool['train'])))
    check_existing_files(old_val, 'Old validation replay')
    baseline_field, _ = evaluate_field(model97, model18, classes97, crops18, field_val, device, args.batch_size)
    baseline_old = evaluate_old(model97, classes97, old_val, device, args.batch_size)
    print('===== BASELINE VALIDATION =====')
    print('Field val:', baseline_field)
    print('Original replay val:', baseline_old)
    save_json(out / 'BASELINE_VALIDATION.json', {'field': baseline_field, 'old': baseline_old,
                                                 'model18': str(ckpt18), 'model97': str(ckpt97)})

    if args.mode == 'test':
        if not args.candidate:
            raise ValueError('Trong --mode test phai truyen --candidate duong_dan_checkpoint')
        with open(out / 'QUYET_DINH_VALIDATION.json', encoding='utf-8') as f:
            decision = json.load(f)
        if not decision.get('approved_for_locked_test'):
            raise ValueError('Validation khong duoc chap nhan; khong danh gia candidate tren Test')
        candidate, names_candidate = load_checkpoint(Path(args.candidate), 97, device)
        if names_candidate != classes97:
            raise ValueError('Class order candidate khac baseline')
        final_candidate = candidate
    else:
        check_existing_files(field_train, 'Field train')
        # Don't materialize old 100k+ image paths: only check sampled training paths per epoch.
        history = []
        best_eligible = None
        best_score = (-1, -1)
        patience_count = 0
        current_stage = None
        teacher, _ = load_checkpoint(ckpt97, 97, device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        for epoch in range(1, args.epochs + 1):
            train_backbone = epoch > args.head_epochs
            if current_stage != train_backbone:
                opt = setup_training(model97, train_backbone, args.lr_head, args.lr_backbone)
                current_stage = train_backbone
            rng = random.Random(args.seed + epoch)
            field_take = choose_field(field_train, rng, args.field_per_epoch)
            old_take = choose_old(old_pool['train'], rng, args.old_per_epoch)
            train_samples = field_take + old_take
            check_existing_files(train_samples, f'Epoch {epoch} train')
            loader = build_loader(train_samples, mapper, device, args.batch_size, True)
            train_stat = train_epoch(model97, teacher, loader, opt, device, device.type == 'cuda', args.kd_weight)
            field_stat, _ = evaluate_field(model97, model18, classes97, crops18, field_val, device, args.batch_size)
            old_stat = evaluate_old(model97, classes97, old_val, device, args.batch_size)
            retention_ok = old_stat['pct'] >= baseline_old['pct'] - args.max_old_drop_pct
            progress = {
                'epoch': epoch, 'stage': 'backbone+head' if train_backbone else 'head',
                **train_stat,
                'field_val_joint_pct': field_stat['joint_pct'],
                'field_val_given_crop_pct': field_stat['given_true_crop_pct'],
                'old_val_acc_pct': old_stat['pct'],
                'retention_ok': retention_ok,
            }
            history.append(progress)
            write_csv(out / 'LICH_SU_FINE_TUNE.csv', history)
            print('Epoch', epoch, '=>', progress, flush=True)
            # Select ONLY on validation, never on Test. Retention is a hard constraint.
            score = (field_stat['joint_correct'], field_stat['given_true_crop_correct'])
            if retention_ok and score > best_score:
                best_score = score
                best_eligible = progress
                torch.save({
                    'model_state_dict': model97.state_dict(),
                    'class_names': classes97,
                    'adaptation': 'field97_v2_locked_train_with_old_replay',
                    'source_checkpoint': str(ckpt97),
                    'epoch': epoch, 'field_val': field_stat, 'old_val': old_stat,
                }, out / 'field97_best_candidate.pth')
                patience_count = 0
            else:
                patience_count += 1
            if epoch > args.head_epochs and patience_count >= args.patience:
                print('Early stopping tren Validation, epoch', epoch)
                break

        approved = bool(best_eligible and
                        best_eligible['field_val_joint_pct'] >= baseline_field['joint_pct'] + args.min_joint_gain_pct and
                        best_eligible['field_val_given_crop_pct'] >= baseline_field['given_true_crop_pct'])
        decision = {
            'approved_for_locked_test': approved,
            'baseline_field_val': baseline_field,
            'baseline_old_val': baseline_old,
            'best_validation': best_eligible,
            'rule': f'Joint gain >= {args.min_joint_gain_pct}pp, disease given crop non-decreasing, original old val drop <= {args.max_old_drop_pct}pp',
            'candidate_path': str(out / 'field97_best_candidate.pth') if best_eligible else None,
            'no_auto_deploy': True,
        }
        save_json(out / 'QUYET_DINH_VALIDATION.json', decision)
        print('===== VALIDATION DECISION =====')
        print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
        if not approved:
            print('NOT APPROVED: giu model cu; khong su dung field97 checkpoint cho Streamlit.')
            return
        final_candidate, cand_classes = load_checkpoint(out / 'field97_best_candidate.pth', 97, device)
        assert cand_classes == classes97

    # Locked Test is used ONLY once after validation-based selection; no tuning here.
    baseline_test, baseline_pred = evaluate_field(model97 if args.mode == 'test' else load_checkpoint(ckpt97, 97, device)[0],
                                                  model18, classes97, crops18, field_test, device, args.batch_size, True)
    trial_test, trial_pred = evaluate_field(final_candidate, model18, classes97, crops18,
                                           field_test, device, args.batch_size, True)
    if len(baseline_pred) != len(trial_pred):
        raise ValueError('Mau so test khong khop!')
    comparison = []
    for old, new in zip(baseline_pred, trial_pred):
        if old['path'] != new['path'] or old['true_class'] != new['true_class']:
            raise ValueError('Test alignment error')
        comparison.append({
            'path': old['path'], 'true_class': old['true_class'],
            'pred_crop': old['pred_crop'],
            'baseline_class': old['pred_class'], 'baseline_correct': old['correct_joint'],
            'new_class': new['pred_class'], 'new_correct': new['correct_joint'],
            'new_given_true_crop': new['pred_given_true_crop'],
        })
    write_csv(out / 'SO_SANH_TEST_DA_KHOA.csv', comparison)
    write_csv(out / 'KET_QUA_THEO_LOP_TEST.csv', summary_by_group(trial_pred))
    disease_compare = defaultdict(list)
    for row in comparison:
        disease_compare[row['true_class']].append(row)
    disease_summary = []
    for name, vals in sorted(disease_compare.items()):
        old_correct = sum(r['baseline_correct'] for r in vals)
        new_correct = sum(r['new_correct'] for r in vals)
        disease_summary.append({'class': name, 'total': len(vals),
                                'old_correct': old_correct, 'new_correct': new_correct,
                                'old_pct': pct(old_correct, len(vals)),
                                'new_pct': pct(new_correct, len(vals)),
                                'delta_correct': new_correct - old_correct})
    write_csv(out / 'SO_SANH_BENH_THEO_NHOM_TEST.csv', disease_summary)
    tomato_rows = [r for r in comparison if r['true_class'].startswith('Tomato_')]
    tomato_summary = {'count': len(tomato_rows),
                      'old_correct': sum(r['baseline_correct'] for r in tomato_rows),
                      'new_correct': sum(r['new_correct'] for r in tomato_rows)}
    summary = {
        'baseline': baseline_test,
        'candidate': trial_test,
        'tomato': tomato_summary,
        'delta_joint_pp': round(trial_test['joint_pct'] - baseline_test['joint_pct'], 2),
        'locked_test_count': len(splits['test']),
        'labeled_test_count': len(field_test),
        'unverified_test_count': len(splits['test']) - len(field_test),
        'model18': str(ckpt18), 'model97_original': str(ckpt97),
        'model97_candidate': str(args.candidate or (out / 'field97_best_candidate.pth')),
        'note': 'Test used only after validation selection. Test result cannot promise 100% accuracy.',
        'auto_deployed': False,
        'candidate_better_on_locked_test': trial_test['joint_correct'] > baseline_test['joint_correct'],
    }
    save_json(out / 'TONG_KET_TEST_V2.json', summary)
    print('===== KET QUA TEST DA KHOA =====')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Khong ghi de original checkpoint va khong tu dong sua app.')


if __name__ == '__main__':
    main()
