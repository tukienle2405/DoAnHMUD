@echo off
chcp 65001 > nul
cd /d "%~dp0"
py -3 -u phan_tich_loi_v6.py
if errorlevel 1 (
 echo.
 echo CO LOI - xem thong bao tren.
 pause
 exit /b 1
)
echo.
echo HOAN TAT. Dang mo HTML...
start "" "%~dp0KET_QUA_RASOAT_V6\XEM_ANH_DU_DOAN_SAI.html"
pause
