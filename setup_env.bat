@echo off
echo Setting up Python 3.10.9 virtual environment...

:: Check if Python 3.10.9 is installed
python --version 2>NUL | findstr /C:"Python 3.10" >NUL
if %ERRORLEVEL% NEQ 0 (
    echo Python 3.10.x is not installed or not in PATH.
    echo Please install Python 3.10.9 from https://www.python.org/downloads/release/python-3109/
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt

echo Environment setup complete!
echo To activate the environment, run: venv\Scripts\activate.bat
