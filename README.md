# Game Prediction Dashboard

A Streamlit dashboard that predicts game outcomes using machine learning. This application uses Supabase for data storage.

## Features

- Real-time game predictions with confidence levels
- Pattern analysis of historical game data
- Auto-refreshing dashboard
- Cloud storage using Supabase

## Quick Start

### Windows Users
1. **Run the app** (sets up environment automatically):
   - Double-click `run_app.bat`

### Advanced Users

1. **Setup your environment**:
   - **Windows (PowerShell)**: Right-click `setup_env.ps1` → "Run with PowerShell"
   - **Windows (CMD)**: Double-click `setup_env.bat`
   - **Unix/Mac**: Run `bash setup_env.sh` in terminal

2. **Activate the environment**:
   - **Windows (PowerShell)**: `.\venv\Scripts\Activate.ps1`
   - **Windows (CMD)**: `venv\Scripts\activate.bat`
   - **Unix/Mac**: `source venv/bin/activate`

3. **Run the application**:
   - `streamlit run run_dashboard.py`

## Troubleshooting

If you encounter issues with running the scripts or activating the environment, see the detailed guides:
- `POWERSHELL_GUIDE.md` - Detailed PowerShell instructions
- `README_DEPLOY.md` - Full deployment instructions

## Environment Variables

The following environment variables are used:

- `SUPABASE_URL`: URL of your Supabase project
- `SUPABASE_KEY`: API key for Supabase access
- `USE_SUPABASE`: Set to "true" to use Supabase for data storage

## Deployment

This application is configured to run on Streamlit Cloud using data stored in Supabase.
See `README_DEPLOY.md` for detailed deployment instructions.

## Requirements

- Python 3.10.x
- See `requirements.txt` for required packages
