# ChatterBox Voice Cloning Service - PowerShell Setup Script
# چیٹر باکس وائس کلوننگ سروس - پاور شیل سیٹ اپ سکرپٹ

Write-Host "================================" -ForegroundColor Cyan
Write-Host "ChatterBox TTS Service Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Step 1: Create virtual environment
Write-Host "Step 1: Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
    Write-Host "Please make sure Python 3.9+ is installed" -ForegroundColor Red
    pause
    exit 1
}

# Step 2: Activate virtual environment
Write-Host ""
Write-Host "Step 2: Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Step 3: Upgrade pip
Write-Host ""
Write-Host "Step 3: Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Step 4: Install dependencies
Write-Host ""
Write-Host "Step 4: Installing dependencies..." -ForegroundColor Yellow
pip install flask==3.0.0
pip install flask-cors==4.0.0
pip install huggingface-hub==0.19.4
pip install transformers==4.35.0
pip install numpy==1.24.3
pip install scipy==1.11.4
pip install soundfile==0.12.1
pip install python-dotenv==1.0.0

# Step 5: Install PyTorch
Write-Host ""
Write-Host "Step 5: Installing PyTorch (CPU version)..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the service:" -ForegroundColor Cyan
Write-Host "  1. .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  2. python app.py" -ForegroundColor White
Write-Host ""
pause
