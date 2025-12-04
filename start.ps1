# ChatterBox TTS Service - Start Script
# اس سکرپٹ کو چلانے کے لیے: .\start.ps1

Write-Host "================================" -ForegroundColor Cyan
Write-Host "Starting ChatterBox TTS Service" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Check if virtual environment exists
if (-not (Test-Path ".\venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup.ps1 first to install dependencies" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run this command: .\setup.ps1" -ForegroundColor White
    pause
    exit 1
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Check if Flask is installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$flaskInstalled = python -c "import flask; print('OK')" 2>$null
if ($flaskInstalled -ne "OK") {
    Write-Host "ERROR: Dependencies not installed!" -ForegroundColor Red
    Write-Host "Please run setup.ps1 first" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host ""
Write-Host "Starting service on http://localhost:5001 ..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Gray
Write-Host ""

# Start the Flask app
python app.py
