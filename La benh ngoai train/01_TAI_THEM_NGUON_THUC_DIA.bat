@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
%PY% -m pip install Pillow
%PY% tai_anh_18_cay_97_lop.py --mendeley --target 5
pause
