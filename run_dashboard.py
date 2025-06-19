import streamlit as st
import os
import subprocess
import sys
import pandas as pd
import builtins
from dotenv import load_dotenv
from pathlib import Path

# Function to set up Supabase environment
def setup_supabase_env():
    # Load environment variables from .env file if it exists
    env_file = Path(".env")
    if not env_file.exists():
        # Create .env file with Supabase credentials if it doesn't exist
        with open(env_file, "w") as f:
            f.write("# Supabase credentials\n")
            f.write("SUPABASE_URL=https://rbflzmvyzldbvshbtqvx.supabase.co\n")
            f.write("SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJiZmx6bXZ5emxkYnZzaGJ0cXZ4Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAxMDkyNDIsImV4cCI6MjA2NTY4NTI0Mn0.ndn9vs5boCgAxRWdpdPI8A-t1uQXtIspWNufNpk1g9Y\n")
        print("Created .env file with Supabase credentials")
    
    # Load environment variables
    load_dotenv()
    
    # Ensure the required packages are installed
    try:
        import supabase
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "supabase-py", "python-dotenv"])
        print("Packages installed successfully")
    
    # Initialize Supabase client and helper functions
    try:
        from supabase import create_client
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            print("No Supabase connection available.")
            return False
        
        # Create Supabase client
        client = create_client(url, key)
        
        # Set up helper functions as builtins for use in dashboard.py
        def supabase_fetch_table_data(table_name):
            try:
                # First, get the total count of records
                count_response = client.table(table_name).select("*", count='exact').limit(1).execute()
                total_count = 0
                
                if hasattr(count_response, 'count'):
                    total_count = count_response.count
                elif isinstance(count_response, dict) and 'count' in count_response:
                    total_count = count_response['count']
                    
                print(f"Total records in {table_name}: {total_count}")
                
                # Initialize an empty list to hold all data
                all_data = []
                
                # Define batch size - Supabase typically has a limit of 1000 records per request
                batch_size = 1000
                
                # Calculate how many batches we need
                num_batches = (total_count + batch_size - 1) // batch_size  # Ceiling division
                
                print(f"Fetching data in {num_batches} batches of {batch_size} records each")
                
                # Fetch data in batches
                for batch in range(num_batches):
                    offset = batch * batch_size
                    print(f"Fetching batch {batch + 1}/{num_batches} (offset: {offset})")
                    
                    # Use range for pagination (offset + limit)
                    response = client.table(table_name).select("*").range(offset, offset + batch_size - 1).execute()
                    
                    # Extract data based on response structure
                    batch_data = []
                    if hasattr(response, 'data'):
                        # Direct attribute access (newer supabase-py versions)
                        batch_data = response.data
                    elif isinstance(response, dict) and 'data' in response:
                        # Dictionary access (older versions)
                        batch_data = response['data']
                    else:
                        print(f"Batch {batch + 1}: No data found in response. Response type: {type(response)}")
                        continue
                        
                    # Add batch data to our collection
                    all_data.extend(batch_data)
                    
                print(f"Total records collected: {len(all_data)}")
                
                # Ensure we have data
                if not all_data:
                    print("No data collected from any batch")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(all_data)
                
                # Convert timestamp column to datetime if it exists
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
                return df
            except Exception as e:
                print(f"Error fetching data from Supabase: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return pd.DataFrame()
        
        def supabase_insert_table_data(table_name, data_dict):
            try:
                # Check if this is a prediction and has a period
                if table_name == 'predictions_data' and 'period' in data_dict:
                    # Check if a record with this period already exists
                    period = data_dict['period']
                    existing_response = client.table(table_name).select("id").eq("period", str(period)).execute()
                    
                    existing_data = []
                    if hasattr(existing_response, 'data'):
                        existing_data = existing_response.data
                    elif isinstance(existing_response, dict) and 'data' in existing_response:
                        existing_data = existing_response['data']
                        
                    if existing_data:
                        # Record exists, update it
                        record_id = existing_data[0]['id']
                        print(f"Found existing prediction for period {period}. Updating instead of inserting.")
                        response = client.table(table_name).update(data_dict).eq("id", record_id).execute()
                    else:
                        # No record exists, insert new one
                        print(f"No existing prediction found for period {period}. Inserting new record.")
                        response = client.table(table_name).insert(data_dict).execute()
                else:
                    # For other tables or data without period, just insert
                    response = client.table(table_name).insert(data_dict).execute()
                
                if hasattr(response, 'data') and response.data:
                    print(f"Successfully inserted/updated data in {table_name}")
                    return True
                else:
                    print(f"Failed to insert/update data in Supabase table {table_name}")
                    return False
            except Exception as e:
                print(f"Error inserting/updating data in Supabase: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return False
        
        # Make these functions available as builtins
        builtins.supabase_fetch_table_data = supabase_fetch_table_data
        builtins.supabase_insert_table_data = supabase_insert_table_data
        builtins.supabase_client = client
        
        print("✅ Supabase client initialized and helper functions ready to use")
        return True
    except Exception as e:
        print(f"Error setting up Supabase client: {str(e)}")
        return False

# Main function
def main():
    # Set up Supabase environment
    setup_supabase_env()
    
    # Check if data directory exists (still keep this for compatibility)
    if not os.path.exists("data"):
        os.makedirs("data", exist_ok=True)
        print("Created data directory")
    
    # Check if models directory exists
    if not os.path.exists("models"):
        os.makedirs("models", exist_ok=True)
        print("Created models directory")
    
    # Get the absolute path for the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Tell the user we're using Supabase for data storage
    print("\n======================================================")
    print("USING SUPABASE CLOUD DATABASE FOR DATA STORAGE")
    print("Data will be read from 'wingo_parity_data' table")
    print("Predictions will be saved to 'predictions_data' table")
    print("======================================================\n")
    
    # Run the dashboard
    print("Starting Streamlit Dashboard...")
    command = [sys.executable, "-m", "streamlit", "run", os.path.join(current_dir, "dashboard.py")]
    
    # Pass environment variables and force Supabase usage
    env = os.environ.copy()
    env["USE_SUPABASE"] = "true"  # Always use Supabase for data storage
    
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
    except KeyboardInterrupt:
        print("Dashboard stopped by user")

if __name__ == "__main__":
    main()
