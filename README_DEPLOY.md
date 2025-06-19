# Deployment Instructions for Prediction Dashboard

This guide explains how to set up your environment and deploy the Prediction Dashboard application on Streamlit Cloud.

## Local Setup

### Setting up the Python Environment

#### For Windows Users

1. **Easy Setup (Recommended)**:
   - Simply double-click `run_app.bat` to set up the environment and run the application in one step

2. **PowerShell Setup**:
   - Right-click on `setup_env.ps1` and select "Run with PowerShell"
   - Or open PowerShell and run: `.\setup_env.ps1`
   - Activate the environment: `.\venv\Scripts\Activate.ps1`

3. **Command Prompt Setup**:
   - Double-click on `setup_env.bat` or run it from Command Prompt
   - Activate the environment: `venv\Scripts\activate.bat`

See `POWERSHELL_GUIDE.md` for detailed PowerShell instructions.

#### For Unix/Mac Users

1. Run the setup script:
   ```bash
   bash setup_env.sh
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

### Running the Application Locally

1. **With Environment Activated**:
   - Windows PowerShell: `streamlit run run_dashboard.py`
   - Windows CMD: `streamlit run run_dashboard.py`
   - Unix/Mac: `streamlit run run_dashboard.py`

2. **Using Run Scripts**:
   - Windows PowerShell: `.\run_streamlit.ps1`
   - Windows CMD: `run_streamlit.bat`

### Environment Variables

The application uses the following environment variables stored in the `.env` file:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

These are automatically loaded when the application starts.

## Deploying to Streamlit Cloud

1. Push your code to a GitHub repository
   - Make sure `.env` is included in `.gitignore` to avoid exposing your secrets

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in

3. Click "New app" and connect your GitHub repository

4. Configure the app:
   - Main file path: `run_dashboard.py`
   - Python version: `3.10.9`
   - Advanced settings:
     - Requirements: `requirements-streamlit.txt`

5. Add the following secrets to your Streamlit app (Settings → Secrets):
   ```toml
   [supabase]
   url = "your_supabase_url"
   key = "your_supabase_key"
   
   [app]
   env = "production"
   ```

6. Click "Deploy" and wait for the application to build and launch

## Project Structure

- `dashboard.py`: Main dashboard components and UI
- `data_manager.py`: Data handling and storage functions
- `ml_predictor.py`: Machine learning prediction functionality
- `run_dashboard.py`: Entry point for the Streamlit application
- `requirements.txt`: Python dependencies
- `requirements-streamlit.txt`: Optimized dependencies for Streamlit deployment
- `models/`: Trained machine learning models

## Troubleshooting

If you encounter any issues during deployment:

1. Check the Streamlit logs for error messages
2. Ensure all dependencies are correctly specified in `requirements-streamlit.txt`
3. Verify that your Streamlit secrets are correctly configured
4. Make sure you're using Python 3.10.9 specifically, as the models may depend on this version
