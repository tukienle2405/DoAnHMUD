@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 -u "%~dp0rasoat_train_v9.py"
echo.
echo === V9 ket thuc. Ket qua nam trong KET_QUA_V9_TRAIN ===
pause
