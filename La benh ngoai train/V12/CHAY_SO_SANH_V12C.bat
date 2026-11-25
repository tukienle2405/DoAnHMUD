@echo off
cd /d "%~dp0"
py -3 -u so_sanh_du_doan_v12c.py --mode predict --batch_size 4
pause
