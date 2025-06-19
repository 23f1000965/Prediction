# PowerShell script to run Streamlit application
Write-Host "Deploying Streamlit application..." -ForegroundColor Green

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Run Streamlit
Write-Host "Starting Streamlit server..." -ForegroundColor Green
streamlit run run_dashboard.py

Write-Host "Streamlit application started!" -ForegroundColor Green
