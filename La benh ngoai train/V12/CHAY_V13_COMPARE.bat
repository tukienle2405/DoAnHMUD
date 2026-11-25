@echo off
cd /d "%~dp0"
py -3 -u kiem_tra_chon_cay_v13.py --mode compare --batch_size 4
pause
