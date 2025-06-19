# Streamlit Cloud Deployment Instructions

### 1. Go to Streamlit Cloud Dashboard
- Visit: https://share.streamlit.io/

### 2. Deploy Your App
- Click "New app"
- Connect to your GitHub repository
- Configure:
  - Repository: Your GitHub repo
  - Branch: main
  - **Main file path: dashboard.py**  ← IMPORTANT! Use this file directly now
  - Python version: 3.10.x
  
### 3. Add Secrets
- After deployment starts, click on the three dots menu
- Select "Settings" then "Secrets"
- Add these secrets:
  ```
  [supabase]
  url = "https://rbflzmvyzldbvshbtqvx.supabase.co"
  key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJiZmx6bXZ5emxkYnZzaGJ0cXZ4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAxMDkyNDIsImV4cCI6MjA2NTY4NTI0Mn0.ndn9vs5boCgAxRWdpdPI8A-t1uQXtIspWNufNpk1g9Y"

  [app]
  use_supabase = true
  ```

### 4. Restart the App
- Click on the three dots menu
- Select "Reboot app"

### Troubleshooting
If you see a blank screen:
1. Check the app logs for errors
2. Make sure you're using streamlit_app.py as the main file
3. Verify your secrets are correctly configured
