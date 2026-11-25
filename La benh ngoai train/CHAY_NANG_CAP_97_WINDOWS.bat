@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo =============================================================
echo  CHECK SPLIT + LABELS (SAFE, NO GPU)
echo =============================================================
py -3 -u nang_cap_benh_97_thuc_dia.py --mode audit
if errorlevel 1 goto ERROR

echo.
echo =============================================================
echo  TRAIN 97 DISEASES + REPLAY, VALIDATE, THEN LOCKED TEST
 echo  Original checkpoints are NEVER overwritten.
echo =============================================================
py -3 -u nang_cap_benh_97_thuc_dia.py --mode train --epochs 8 --batch_size 4
if errorlevel 1 goto ERROR

echo.
echo FINISHED. Check D:\DoanHMvUD\training_runs\field97_v2_safe
pause
exit /b 0
:ERROR
echo.
echo ERROR. Copy terminal output to review; no original model changed.
pause
exit /b 1
