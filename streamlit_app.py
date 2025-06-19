import streamlit as st
import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Game Prediction Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Set Supabase environment variables from secrets
try:
    # For Streamlit Cloud
    from streamlit.web.bootstrap import get_config_options
    if hasattr(st, 'secrets') and 'supabase' in st.secrets:
        os.environ['SUPABASE_URL'] = st.secrets['supabase']['url']
        os.environ['SUPABASE_KEY'] = st.secrets['supabase']['key']
except:
    # Local development, use .env file
    pass

# Make sure USE_SUPABASE is set
os.environ['USE_SUPABASE'] = 'true'

# Import the dashboard module - this will run all the dashboard code
try:
    # Import dashboard.py directly - it will execute all its code
    import dashboard
except Exception as e:
    st.error(f"Error loading dashboard: {e}")
    import traceback
    st.code(traceback.format_exc())
