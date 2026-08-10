@echo off
REM ===================================================================
REM  Train all three detection models, independently of any editor.
REM
REM  Double-click this file AFTER closing VS Code, Claude, Discord and
REM  Zalo. Those four together hold about 3.5 GB, and on this 16 GB
REM  machine the DataLoader worker count - which is what the GPU is
REM  actually waiting on - is limited by free RAM, not by VRAM or by
REM  the 20 CPU cores.
REM
REM  The script sizes its own worker count from whatever RAM is free
REM  when it starts, so the more you close, the faster it runs.
REM ===================================================================

title Fruit/Veg Detection - training 3 models
cd /d "%~dp0"

set PYTHON=C:\Users\KUONG\miniconda3\envs\objdet\python.exe
if not exist "%PYTHON%" (
    echo ERROR: python not found at %PYTHON%
    echo Activate the conda env manually and run:
    echo     python scripts\12_train_all.py
    pause
    exit /b 1
)

echo ===================================================================
echo  Training SSD300-VGG16, YOLOv8m and RT-DETR-l, one after another.
echo.
echo  Progress appears below and is also written to runs\logs\.
echo  Leave this window open. Ctrl+C stops training.
echo ===================================================================
echo.

"%PYTHON%" scripts\12_train_all.py %*

echo.
echo ===================================================================
echo  FINISHED - logs in runs\logs\, metrics in reports\results\.
echo  Reopen Claude Code in this folder to continue the analysis.
echo ===================================================================
pause
