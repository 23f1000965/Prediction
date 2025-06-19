# PowerShell Deployment Guide

This document provides step-by-step instructions for setting up and deploying the Prediction Dashboard application using PowerShell on Windows.

## Setting Up the Environment

### Option 1: Using PowerShell Script (Recommended)

1. Right-click on `setup_env.ps1` and select "Run with PowerShell"

   OR

   Open PowerShell in the project directory and run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\setup_env.ps1
   ```

2. Wait for the script to finish setting up the environment

3. To activate the environment, run:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   You should see `(venv)` at the beginning of your PowerShell prompt

### Option 2: Manual Setup

If you prefer to set up the environment manually:

1. Open PowerShell in the project directory

2. Create a virtual environment:
   ```powershell
   python -m venv venv
   ```

3. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. Install the required packages:
   ```powershell
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Using PowerShell Script (Recommended)

1. Right-click on `run_streamlit.ps1` and select "Run with PowerShell"

   OR

   Open PowerShell in the project directory and run:
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\run_streamlit.ps1
   ```

### Option 2: Manual Run

1. Activate the virtual environment (if not already activated):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Run the Streamlit application:
   ```powershell
   streamlit run run_dashboard.py
   ```

### Option 3: Direct Run (No Virtual Environment)

If you have all the required packages installed globally:

1. Right-click on `run_direct.bat` and select "Run"

   OR

   Open PowerShell/Command Prompt in the project directory and run:
   ```
   .\run_direct.bat
   ```

## Troubleshooting

### Execution Policy Issues

If you encounter an error about execution policy:

```
File cannot be loaded because running scripts is disabled on this system.
```

Run PowerShell as Administrator and execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```

Or for the current process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Module Not Found Errors

If you get errors about missing modules:

1. Make sure you've activated the virtual environment
2. Try reinstalling the packages:
   ```powershell
   pip install -r requirements.txt --force-reinstall
   ```

### Python Version Issues

This application is designed to work with Python 3.10.x. To check your Python version:

```powershell
python --version
```

If you need to install Python 3.10.9, download it from:
https://www.python.org/downloads/release/python-3109/
