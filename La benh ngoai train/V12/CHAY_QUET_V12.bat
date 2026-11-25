@echo off
chcp 65001 >nul
cd /d "%~dp0"
py -3 -u quet_anh_cu_cho_v12.py
pause
