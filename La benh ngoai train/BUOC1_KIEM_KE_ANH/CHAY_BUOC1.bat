@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 -u kiem_ke_anh_buoc1.py
pause
