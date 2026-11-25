# -*- coding: utf-8 -*-
"""1 lệnh: audit -> fusion VAL -> train crop18 VAL; KHÔNG chạy Test hoặc sửa App."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nang_cap_benh_97_thuc_dia as v2

HOME = Path(__file__).resolve().parent
ROOT = v2.find_root()


def run(label, command):
    print(f'\n===== {label} =====', flush=True)
    print(' '.join(map(str, command)), flush=True)
    r = subprocess.run(command, cwd=HOME)
    if r.returncode:
        raise SystemExit(f'BUOC {label} LOI (exit {r.returncode}). Khong chuyen sang buoc khac.')


def main():
    python = sys.executable
    run('AUDIT NHAN CAY', [python, '-u', str(HOME/'nang_cap_cay_18_v3.py'), '--mode', 'audit'])
    run('PHOI HOP DIEM CHI TREN VALIDATION', [python, '-u', str(HOME/'kiem_tra_fusion_validation_v3.py'), '--batch_size', '8'])
    run('FINE-TUNE 18 CAY CHI TREN TRAIN', [python, '-u', str(HOME/'nang_cap_cay_18_v3.py'),
                                           '--mode', 'train', '--epochs', '6', '--batch_size', '4'])
    fpath = ROOT / 'training_runs' / 'field97_v3_fusion_val_only' / 'TONG_KET_FUSION_VALIDATION.json'
    cpath = ROOT / 'training_runs' / 'field18_v3_safe' / 'QUYET_DINH_V3_VALIDATION.json'
    summary = {
        'fusion_validated': None,
        'crop_training_validated': None,
        'no_test_touched': True,
        'streamlit_not_modified': True,
        'old_checkpoints_not_overwritten': True,
        'next_required': 'Independent image holdout with verified plant and disease labels, including missing crops',
    }
    if fpath.is_file():
        f = json.loads(fpath.read_text(encoding='utf-8'))
        summary['fusion_validated'] = f.get('approved_for_further_independent_verification')
        summary['fusion_chosen'] = f.get('chosen')
    if cpath.is_file():
        c = json.loads(cpath.read_text(encoding='utf-8'))
        summary['crop_training_validated'] = c.get('approved_for_new_independent_verification')
        summary['crop_chosen'] = c.get('best')
    final = ROOT / 'training_runs' / 'TOM_TAT_V3_VALIDATION.json'
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n===== KET QUA V3 (KHONG DUNG TEST) =====')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('Bao cao:', final, flush=True)
    print('Khong thay doi Streamlit; chua du dieu kien cong bo 18 cay/97 benh dung ngoai thuc te.', flush=True)


if __name__ == '__main__':
    main()
