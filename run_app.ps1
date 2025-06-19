# PowerShell script to set up and run the Streamlit application
param(
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"

function Write-Colored {
    param(
        [string]$Text,
        [string]$Color = "White"
    )
    Write-Host $Text -ForegroundColor $Color
}

# Header
Write-Colored "=====================================================" "Cyan"
Write-Colored "   Prediction Dashboard - Setup and Run Script       " "Green"
Write-Colored "=====================================================" "Cyan"

# Check if virtual environment exists
$venvExists = Test-Path -Path ".\venv\Scripts\activate.ps1"

if (-not $venvExists -and -not $SkipSetup) {
    Write-Colored "`nSetting up Python environment..." "Yellow"
    
    # Check Python version
    try {
        $pythonVersion = python --version 2>&1
        Write-Colored "Detected: $pythonVersion" "Gray"
        
        if (-not ($pythonVersion -like "*Python 3*")) {
            Write-Colored "WARNING: Python 3.x is recommended for this application." "Yellow"
            $continue = Read-Host "Continue anyway? (y/n)"
            if ($continue -ne "y") {
                Write-Colored "Setup aborted." "Red"
                exit 1
            }
        }
    }
    catch {
        Write-Colored "ERROR: Python is not installed or not in PATH." "Red"
        Write-Colored "Please install Python 3.10.x from https://www.python.org/downloads/" "Yellow"
        exit 1
    }
    
    # Create virtual environment
    Write-Colored "`nCreating virtual environment..." "Green"
    try {
        python -m venv venv
    }
    catch {
        Write-Colored "ERROR: Failed to create virtual environment." "Red"
        Write-Colored "Error details: $_" "Red"
        exit 1
    }
    
    # Activate virtual environment
    Write-Colored "`nActivating virtual environment..." "Green"
    try {
        & .\venv\Scripts\activate.ps1
    }
    catch {
        Write-Colored "ERROR: Failed to activate virtual environment." "Red"
        Write-Colored "Error details: $_" "Red"
        exit 1
    }
    
    # Install requirements
    Write-Colored "`nInstalling requirements..." "Green"
    try {
        pip install --upgrade pip
        pip install -r requirements.txt
    }
    catch {
        Write-Colored "ERROR: Failed to install requirements." "Red"
        Write-Colored "Error details: $_" "Red"
        exit 1
    }
    
    Write-Colored "`nEnvironment setup complete!" "Green"
} 
elseif ($SkipSetup) {
    Write-Colored "`nSkipping environment setup as requested." "Yellow"
}
else {
    Write-Colored "`nVirtual environment already exists." "Green"
    
    # Activate virtual environment
    Write-Colored "Activating virtual environment..." "Green"
    try {
        & .\venv\Scripts\activate.ps1
    }
    catch {
        Write-Colored "ERROR: Failed to activate virtual environment." "Red"
        Write-Colored "Error details: $_" "Red"
        exit 1
    }
}

# Run Streamlit
Write-Colored "`nStarting Streamlit server..." "Cyan"
try {
    streamlit run run_dashboard.py
}
catch {
    Write-Colored "ERROR: Failed to start Streamlit server." "Red"
    Write-Colored "Error details: $_" "Red"
    exit 1
}

Write-Colored "`nStreamlit application stopped." "Yellow"
