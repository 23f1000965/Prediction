@echo off
echo Updating Streamlit deployment settings...

:: Create or update .streamlit/config.toml if needed
if not exist .streamlit\config.toml (
    echo Creating Streamlit configuration...
    if not exist .streamlit mkdir .streamlit
    copy .streamlit\config.toml.example .streamlit\config.toml 2>NUL
)

:: Create or update .streamlit/secrets.toml if needed
if not exist .streamlit\secrets.toml (
    echo Creating Streamlit secrets...
    echo [supabase] > .streamlit\secrets.toml
    echo url = "%SUPABASE_URL%" >> .streamlit\secrets.toml
    echo key = "%SUPABASE_KEY%" >> .streamlit\secrets.toml
    echo [app] >> .streamlit\secrets.toml
    echo env = "production" >> .streamlit\secrets.toml
)

echo Streamlit deployment settings updated!
echo To deploy on Streamlit Cloud:
echo 1. Push your code to GitHub
echo 2. Go to https://streamlit.io/cloud
echo 3. Connect your GitHub repository
echo 4. Set the main file as 'run_dashboard.py'
echo 5. Set the Python version to 3.10.9
echo 6. Add the secrets from your .env file to the Streamlit Cloud secrets
