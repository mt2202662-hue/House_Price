@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python eda.py
echo.
echo EDA finished. Review the outputs folder.
pause
