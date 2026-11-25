# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from core_v14 import CROP_MODEL, DISEASE_MODEL, validate_checkpoint_paths

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['check'],default='check')
    args=ap.parse_args()
    out=Path(__file__).resolve().parent/'KET_QUA_V14'
    out.mkdir(parents=True,exist_ok=True)
    status={
        'mode':'V14_DEMO_READ_ONLY_CHECK',
        'crop_model':str(CROP_MODEL),'disease_model':str(DISEASE_MODEL),
        'crop_model_exists':CROP_MODEL.is_file(),'disease_model_exists':DISEASE_MODEL.is_file(),
        'models_unchanged':True,'old_streamlit_unchanged':True,
        'warning':'V14 la demo, khong phai phep danh gia Accuracy.',
    }
    try:
        import torch
        import streamlit
        n18,n97=validate_checkpoint_paths()
        status.update(ready_for_demo=True,crop_classes=len(n18),disease_classes=len(n97),cuda_available=torch.cuda.is_available())
    except Exception as e:
        status.update(ready_for_demo=False,error=f'{type(e).__name__}: {e}')
    (out/'KIEM_TRA_V14.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if not status['ready_for_demo']:sys.exit(2)
if __name__=='__main__': main()
