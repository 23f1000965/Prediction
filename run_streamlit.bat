@echo off
echo Deploying Streamlit application...

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run Streamlit
echo Starting Streamlit server...
streamlit run run_dashboard.py

echo Streamlit application started!
