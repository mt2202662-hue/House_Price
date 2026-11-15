@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python train.py
echo.
echo Model training finished. Review outputs\model_comparison.csv and outputs\validation_metrics.json.
pause
