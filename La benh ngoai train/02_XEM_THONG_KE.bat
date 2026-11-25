@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
%PY% tai_anh_18_cay_97_lop.py --report --target 5
start "" "D:\DoanHMvUD\external_field_pool_v2\BAO_CAO_97_LOP.csv"
pause
