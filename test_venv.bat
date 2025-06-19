@echo off
echo Testing virtual environment activation...

:: Check if venv exists
if not exist venv\Scripts\activate.bat (
    echo Virtual environment not found!
    echo Please run setup_env.bat first.
    exit /b 1
)

:: Try to activate
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Check if activation was successful
if %ERRORLEVEL% NEQ 0 (
    echo Failed to activate virtual environment!
    exit /b 1
)

:: Print Python path to verify
echo Virtual environment activated successfully!
echo Python path:
where python

:: Deactivate
echo Deactivating virtual environment...
call venv\Scripts\deactivate.bat

echo Test completed successfully!
