@echo off
cd /d "%~dp0"
py -3 -u train_v11_97_safe.py --mode train --epochs 2 --batch_size 8
pause
