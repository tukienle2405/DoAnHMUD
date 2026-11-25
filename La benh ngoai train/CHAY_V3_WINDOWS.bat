@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 -u CHAY_V3_TU_DONG.py
set CODE=%ERRORLEVEL%
echo.
echo ===== V3 ket thuc voi ma %CODE% =====
pause
exit /b %CODE%
