# PowerShell script to set up Python 3.10.9 virtual environment
Write-Host "Setting up Python 3.10.9 virtual environment..." -ForegroundColor Green

# Check if Python 3.10.9 is installed
$pythonVersion = python --version 2>&1
if (-not ($pythonVersion -like "*Python 3.10*")) {
    Write-Host "Python 3.10.x is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10.9 from https://www.python.org/downloads/release/python-3109/" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Green
python -m venv venv

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Green
& .\venv\Scripts\python.exe -m pip install --upgrade pip
& .\venv\Scripts\pip.exe install -r requirements.txt

Write-Host "Environment setup complete!" -ForegroundColor Green
Write-Host "To activate the environment in PowerShell, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
