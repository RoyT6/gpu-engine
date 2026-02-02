@echo off
REM GPU Runner Script for ViewerDBX (Windows batch file)
REM Usage: run_gpu.bat script.py [args...]

REM Check if running in WSL is needed
echo ViewerDBX GPU Runner
echo ====================
echo.

REM For Windows native execution with CUDA
set CUDA_VISIBLE_DEVICES=0
set PYTHONUNBUFFERED=1

REM Run the script
python %*
