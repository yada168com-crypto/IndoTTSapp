@echo off
chcp 65001 >nul
set "BASE_DIR=%~dp0"
set "BASE_DIR=%BASE_DIR:~0,-1%"
set "PYTHONIOENCODING=utf-8"

echo ==============================================
echo Indonesian Qwen3-TTS Runner
echo ==============================================

set "CONDA_BASE=%USERPROFILE%\miniconda3"
if not exist "%CONDA_BASE%\Scripts\activate.bat" (
    echo [ERROR] Could not find conda at "%CONDA_BASE%".
    echo Edit CONDA_BASE in this .bat file if miniconda/anaconda is installed elsewhere.
    pause
    exit /b 1
)

call "%CONDA_BASE%\Scripts\activate.bat" qwen3-tts
if errorlevel 1 (
    echo [ERROR] Failed to activate conda environment "qwen3-tts".
    echo Create it first - see README.md for setup steps.
    pause
    exit /b 1
)

python "%BASE_DIR%\run_indo_tts.py" %*

echo.
pause
