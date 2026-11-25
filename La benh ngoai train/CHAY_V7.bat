@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ===== V7 RA SOAT TU DONG =====
py -3 -u "%~dp0rasoat_v7.py"
if errorlevel 1 (
 echo.
 echo Neu loi thieu Pillow, chay: py -3 -m pip install Pillow
)
echo.
echo Mo thu muc: %~dp0KET_QUA_V7_TU_DONG
pause
