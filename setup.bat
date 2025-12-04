@echo off
REM ChatterBox Voice Cloning Setup Script
REM Run this file by double-clicking or from PowerShell

echo ================================
echo ChatterBox TTS Service Setup
echo ================================
echo.

cd /d "%~dp0"

echo Step 1: Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Please make sure Python 3.9+ is installed
    pause
    exit /b 1
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Step 4: Installing dependencies...
pip install flask==3.0.0
pip install flask-cors==4.0.0
pip install huggingface-hub==0.19.4
pip install transformers==4.35.0
pip install numpy==1.24.3
pip install scipy==1.11.4
pip install soundfile==0.12.1
pip install python-dotenv==1.0.0

echo.
echo Step 5: Installing PyTorch (CPU version)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo.
echo ================================
echo Setup Complete!
echo ================================
echo.
echo To start the service:
echo   1. Open PowerShell in this folder
echo   2. Run: .\venv\Scripts\Activate.ps1
echo   3. Run: python app.py
echo.
pause
