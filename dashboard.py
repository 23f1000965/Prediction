import streamlit as st
import builtins

# Page config is set in streamlit_app.py, do not set it again here
# Otherwise it will cause a StreamlitAPIException

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import datetime
from ml_predictor import GamePredictor
from data_manager import add_new_game_data
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Always use Supabase for data storage
USE_SUPABASE = True

# Define table names in Supabase
GAME_DATA_TABLE = 'wingo_parity_data'
PREDICTION_DATA_TABLE = 'predictions_data'

# Define CSV file paths (only as fallback if Supabase connection fails)
GAME_DATA_CSV = 'data/wingo_parity_data.csv'
PREDICTION_DATA_CSV = 'data/predictions_data.csv'

# Import Supabase client using the latest package structure
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    print("✅ Supabase package successfully imported")
except ImportError as e:
    print(f"Error importing Supabase: {str(e)}")
    SUPABASE_AVAILABLE = False

# Initialize Supabase client with proper error handling
@st.cache_resource
def get_supabase_client():
    """Initialize and return the Supabase client"""
    try:
        # First try to get credentials from Streamlit secrets
        if hasattr(st, 'secrets') and 'supabase' in st.secrets:
            url = st.secrets['supabase']['url']
            key = st.secrets['supabase']['key']
            print("✅ Found Supabase credentials in Streamlit secrets")
        else:
            # Fall back to environment variables
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            print("⚠️ Supabase URL or key not found in environment variables or secrets.")
            st.warning("Supabase URL or key not found. Using local data.")
            return None
            
        print(f"Connecting to Supabase URL: {url}")
        # Create Supabase client with explicit timeout settings
        client = create_client(url, key)
        
        # Test the connection with a simple query
        try:
            # Fix: Using proper PostgREST syntax for count
            response = client.table(GAME_DATA_TABLE).select("*", count="exact").execute()
            print(f"✅ Supabase connection test successful: {response}")
            st.sidebar.success("✅ Connected to Supabase cloud database")
            return client
        except Exception as query_e:
            print(f"⚠️ Supabase connection test failed: {str(query_e)}")
            st.warning(f"Failed to query Supabase: {str(query_e)}. Using local data.")
            return None
        except Exception as query_e:
            print(f"⚠️ Supabase connection test failed: {str(query_e)}")
            st.warning(f"Failed to query Supabase: {str(query_e)}. Using local data.")
            return None
    except Exception as e:
        print(f"⚠️ Error initializing Supabase client: {str(e)}")
        st.warning(f"Failed to initialize Supabase client: {str(e)}. Using local data.")
        return None

# Initialize Supabase client
supabase_client = get_supabase_client()
SUPABASE_READY = supabase_client is not None

# Display warning only if Supabase is explicitly requested but not available
if not SUPABASE_READY and USE_SUPABASE:
    st.sidebar.warning("Supabase client not ready. Will use local files as fallback.")

# Functions to load and prepare data
def load_data(data_file=None):
    """Load game data from Supabase first, fallback to CSV only if needed"""
    # Always try to load from Supabase first
    if USE_SUPABASE and SUPABASE_READY:
        try:
            # Use the initialized Supabase client
            df = load_game_data_from_supabase()
            if not df.empty:
                print(f"Successfully loaded {len(df)} records from Supabase table {GAME_DATA_TABLE}")
                return df
            else:
                print("No data found in Supabase table. Checking local CSV...")
        except Exception as e:
            print(f"Error loading from Supabase: {str(e)}. Falling back to CSV file.")
            st.warning(f"Error loading from Supabase: {str(e)}. Falling back to CSV file.")
    
    # Fallback to CSV only if Supabase failed
    if data_file and os.path.exists(data_file):
        try:
            df = pd.read_csv(
                data_file,
                dtype={'period': str, 'number': str, 'color': str, 'price': float}
            )
            
            # Convert timestamp column to datetime if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            st.warning("⚠️ Using local CSV file instead of Supabase. This is not recommended for cloud deployment.")
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    else:
        st.error(f"No data available. Please add game data to Supabase table '{GAME_DATA_TABLE}'.")
        return None

# Functions to load data from Supabase
def load_game_data_from_supabase():
    """Load game data from Supabase wingo_parity_data table"""
    try:
        if supabase_client is None:
            print("No Supabase client available for loading game data")
            return pd.DataFrame()
            
        # Use Supabase client to fetch data with pagination
        print(f"Querying Supabase table: {GAME_DATA_TABLE}")
        
        # First, get the total count of records
        count_response = supabase_client.table(GAME_DATA_TABLE).select("*", count='exact').limit(1).execute()
        total_count = 0
        
        if hasattr(count_response, 'count'):
            total_count = count_response.count
        elif isinstance(count_response, dict) and 'count' in count_response:
            total_count = count_response['count']
            
        print(f"Total records in table: {total_count}")
        
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
            response = supabase_client.table(GAME_DATA_TABLE).select("*").range(offset, offset + batch_size - 1).execute()
            
            # Extract data based on response structure
            batch_data = []
            if hasattr(response, 'data'):
                # Direct attribute access (newer supabase-py versions)
                batch_data = response.data
                print(f"Batch {batch + 1}: Found {len(batch_data)} records via attribute access")
            elif isinstance(response, dict) and 'data' in response:
                # Dictionary access (older versions)
                batch_data = response['data']
                print(f"Batch {batch + 1}: Found {len(batch_data)} records via dictionary access")
            else:
                print(f"Batch {batch + 1}: No data found in response. Response type: {type(response)}")
                continue
                
            # Add batch data to our collection
            all_data.extend(batch_data)
            
            print(f"Total records collected so far: {len(all_data)}")
        
        # Ensure we have data
        if not all_data:
            print("No data collected from any batch")
            return pd.DataFrame()
            
        # Debug first record
        if all_data:
            print(f"First record: {all_data[0]}")
        else:
            print("No data to show first record")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        print(f"Created DataFrame with shape {df.shape} and columns {df.columns.tolist()}")
        print(f"Successfully fetched all {len(df)} records from Supabase!")
        
        # Convert data types
        if 'number' in df.columns:
            df['number'] = pd.to_numeric(df['number'], errors='coerce')
            
        if 'price' in df.columns:
            # Price can be either numeric or string, keep as is
            pass
            
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        return df
    except Exception as e:
        print(f"Error in load_game_data_from_supabase: {str(e)}")
        import traceback
        print(traceback.format_exc())
        st.warning(f"Failed to load data from Supabase: {str(e)}")
        return pd.DataFrame()

def load_prediction_data_from_supabase():
    """Load prediction data from Supabase predictions_data table"""
    try:
        if supabase_client is None:
            print("No Supabase client available for loading prediction data")
            return pd.DataFrame()
            
        # Use Supabase client to fetch data with pagination
        print(f"Querying Supabase table: {PREDICTION_DATA_TABLE}")
        
        # First, get the total count of records
        count_response = supabase_client.table(PREDICTION_DATA_TABLE).select("*", count='exact').limit(1).execute()
        total_count = 0
        
        if hasattr(count_response, 'count'):
            total_count = count_response.count
        elif isinstance(count_response, dict) and 'count' in count_response:
            total_count = count_response['count']
            
        print(f"Total prediction records in table: {total_count}")
        
        # Initialize an empty list to hold all data
        all_data = []
        
        # Define batch size - Supabase typically has a limit of 1000 records per request
        batch_size = 1000
        
        # Calculate how many batches we need
        num_batches = (total_count + batch_size - 1) // batch_size  # Ceiling division
        
        print(f"Fetching prediction data in {num_batches} batches of {batch_size} records each")
        
        # Fetch data in batches
        for batch in range(num_batches):
            offset = batch * batch_size
            print(f"Fetching prediction batch {batch + 1}/{num_batches} (offset: {offset})")
            
            # Use range for pagination (offset + limit)
            response = supabase_client.table(PREDICTION_DATA_TABLE).select("*").range(offset, offset + batch_size - 1).execute()
            
            # Extract data based on response structure
            batch_data = []
            if hasattr(response, 'data'):
                # Direct attribute access (newer supabase-py versions)
                batch_data = response.data
                print(f"Prediction batch {batch + 1}: Found {len(batch_data)} records via attribute access")
            elif isinstance(response, dict) and 'data' in response:
                # Dictionary access (older versions)
                batch_data = response['data']
                print(f"Prediction batch {batch + 1}: Found {len(batch_data)} records via dictionary access")
            else:
                print(f"Prediction batch {batch + 1}: No data found in response. Response type: {type(response)}")
                continue
                
            # Add batch data to our collection
            all_data.extend(batch_data)
            
            print(f"Total prediction records collected so far: {len(all_data)}")
        
        # Ensure we have data
        if not all_data:
            print("No prediction data collected from any batch")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        print(f"Created prediction DataFrame with shape {df.shape}")
        print(f"Successfully fetched all {len(df)} prediction records from Supabase!")
        print(f"Created prediction DataFrame with shape {df.shape}")
        
        # Process timestamps if they exist
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
        return df
    except Exception as e:
        print(f"Error loading prediction data from Supabase: {str(e)}")
        st.warning(f"Error loading prediction data from Supabase: {str(e)}. Will return empty DataFrame.")
        return pd.DataFrame()

# Function to save prediction data to Supabase or CSV file
def save_prediction_data(period, predicted_number=None, predicted_color=None, 
                         number_confidence=None, color_confidence=None, 
                         number_recommendation=None, color_recommendation=None,
                         actual_number=None, actual_color=None, correct_number=None, correct_color=None,
                         timestamp=None):
    """
    Save prediction data to the predictions CSV file.
    
    Args:
        period: Game period ID that the prediction is for
        predicted_number: Predicted number from ML model
        predicted_color: Predicted color from ML model
        number_confidence: Confidence percentage for number prediction
        color_confidence: Confidence percentage for color prediction
        number_recommendation: Recommendation level for number prediction
        color_recommendation: Recommendation level for color prediction
        timestamp: Optional timestamp (defaults to current time)
    
    Returns:
        Boolean indicating success
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
        
    # Create data dictionary with properly formatted timestamp
    data = {
        'period': str(period),
        'timestamp': timestamp.isoformat() if isinstance(timestamp, (datetime.datetime, pd.Timestamp)) else timestamp
    }
    
    # Add prediction data
    if predicted_number is not None:
        data['predicted_number'] = str(predicted_number)
    if predicted_color is not None:
        data['predicted_color'] = str(predicted_color)
    if number_confidence is not None:
        data['number_confidence'] = str(number_confidence)
    if color_confidence is not None:
        data['color_confidence'] = str(color_confidence)
    if number_recommendation is not None:
        data['number_recommendation'] = str(number_recommendation)
    if color_recommendation is not None:
        data['color_recommendation'] = str(color_recommendation)
        
    # Add actual results data if available
    if actual_number is not None:
        data['actual_number'] = str(actual_number)
    if actual_color is not None:
        data['actual_color'] = str(actual_color)
    if correct_number is not None:
        data['correct_number'] = str(correct_number)
    if correct_color is not None:
        data['correct_color'] = str(correct_color)
    
    # Always save to Supabase, only fall back to CSV if there's an error
    if USE_SUPABASE and SUPABASE_READY and supabase_client is not None:
        try:
            # Convert boolean strings to actual booleans for Supabase
            if 'correct_number' in data and isinstance(data['correct_number'], str):
                data['correct_number'] = data['correct_number'].lower() == 'true'
            if 'correct_color' in data and isinstance(data['correct_color'], str):
                data['correct_color'] = data['correct_color'].lower() == 'true'
                
            # Convert datetime objects and confidence values for Supabase
            json_safe_data = {}
            for key, value in data.items():
                if isinstance(value, (datetime.datetime, pd.Timestamp)):
                    # Format datetime objects as ISO format strings
                    json_safe_data[key] = value.isoformat()
                elif key in ['number_confidence', 'color_confidence'] and value is not None:
                    # Remove percentage symbols and convert to float for confidence values
                    try:
                        if isinstance(value, str) and '%' in value:
                            # Remove percentage sign and convert to float
                            clean_value = value.replace('%', '').strip()
                            json_safe_data[key] = float(clean_value)
                        else:
                            # Try direct conversion to float
                            json_safe_data[key] = float(value)
                    except (ValueError, TypeError):
                        # Fall back to original value if conversion fails
                        print(f"Warning: Could not convert {key}={value} to a number")
                        json_safe_data[key] = value
                else:
                    # Keep other values as is
                    json_safe_data[key] = value
            
            # Check if a prediction for this period already exists
            try:
                # First try to find if this period already has a prediction
                existing_response = supabase_client.table(PREDICTION_DATA_TABLE).select("id").eq("period", str(period)).execute()
                existing_data = []
                
                if hasattr(existing_response, 'data'):
                    existing_data = existing_response.data
                elif isinstance(existing_response, dict) and 'data' in existing_response:
                    existing_data = existing_response['data']
                    
                if existing_data:
                    # Record exists, so update it instead of inserting
                    record_id = existing_data[0]['id']
                    print(f"Found existing prediction for period {period} with ID {record_id}. Updating instead of inserting.")
                    response = supabase_client.table(PREDICTION_DATA_TABLE).update(json_safe_data).eq("id", record_id).execute()
                    print(f"Supabase update response: {response}")
                else:
                    # No existing record, do an insert
                    print(f"No existing prediction found for period {period}. Inserting new record.")
                    print(f"Saving prediction to Supabase table {PREDICTION_DATA_TABLE}: {json_safe_data}")
                    response = supabase_client.table(PREDICTION_DATA_TABLE).insert(json_safe_data).execute()
                    print(f"Supabase insert response: {response}")
            except Exception as check_e:
                # If the check fails, try a direct upsert
                print(f"Error checking for existing record: {str(check_e)}. Attempting upsert.")
                response = supabase_client.table(PREDICTION_DATA_TABLE).upsert(json_safe_data).execute()
                print(f"Supabase upsert response: {response}")
            
            if hasattr(response, 'data') and response.data:
                st.success(f"Prediction saved to Supabase table '{PREDICTION_DATA_TABLE}'!")
                return True
            else:
                print(f"Error saving to Supabase: No data in response")
                # Fall back to CSV in case of errors
                st.warning(f"Error saving to Supabase. Saving to local CSV as fallback.")
                return save_data_to_csv(data, PREDICTION_DATA_CSV)
        except Exception as e:
            print(f"Error saving to Supabase: {str(e)}")
            # Fall back to CSV in case of errors
            st.warning(f"Error saving to Supabase: {str(e)}. Saving to local CSV as fallback.")
            return save_data_to_csv(data, PREDICTION_DATA_CSV)
    else:
        # Supabase client not available - use CSV as last resort
        st.error("Supabase connection not available. Saving to local CSV file.")
        return save_data_to_csv(data, PREDICTION_DATA_CSV)

# Function to save data to CSV file
def save_data_to_csv(data_dict, file_path, create_if_missing=True, use_sequential_order=False):
    """
    Save new data to CSV file.
    
    Args:
        data_dict: Dictionary containing the new data entry
        file_path: Path to the CSV file
        create_if_missing: Create directory if it doesn't exist
        use_sequential_order: If True, periods will be sorted in ascending order
            instead of newest-first default
    """
    try:
        # Check if file exists
        if os.path.exists(file_path):
            # Read existing data
            try:
                existing_df = pd.read_csv(
                    file_path,
                    dtype={'period': str, 'number': str, 'color': str, 'price': float}
                )
            except Exception:
                # Fallback with minimal processing
                existing_df = pd.read_csv(file_path, dtype=object)
                
            # Create new dataframe with the new data
            new_entry = pd.DataFrame([data_dict])
            
            # PREDICTION_DATA_CSV should use sequential ordering
            if file_path == PREDICTION_DATA_CSV or use_sequential_order:
                # Check if entry already exists by period
                if 'period' in existing_df.columns and 'period' in new_entry.columns:
                    # Remove any existing entries with same period as new entry
                    existing_period = new_entry['period'].iloc[0]
                    existing_df = existing_df[existing_df['period'] != existing_period]
                    
                # Combine data and sort by period in ascending order
                updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
                if 'period' in updated_df.columns:
                    updated_df['period'] = updated_df['period'].astype(str)
                    updated_df = updated_df.sort_values(by='period', ascending=True)
            else:
                # For other files like game data, keep newest-first order
                updated_df = pd.concat([new_entry, existing_df], ignore_index=True)
        else:
            # Create new file with just this entry
            updated_df = pd.DataFrame([data_dict])
            
            # Create directory if it doesn't exist
            if create_if_missing:
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
            # Sort by period if it's the prediction data file
            if file_path == PREDICTION_DATA_CSV or use_sequential_order:
                if 'period' in updated_df.columns:
                    updated_df['period'] = updated_df['period'].astype(str)
                    updated_df = updated_df.sort_values(by='period', ascending=True)
        
        # Save the updated dataframe back to CSV
        updated_df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving data to CSV: {str(e)}")
        return False
        
# Create AutoDataCollector class here directly to avoid import issues
class AutoDataCollector:
    """Class to automatically collect and save game data"""
    
    def __init__(self, log_file="data_collector.log"):
        """Initialize data collector"""
        self.last_processed_period = None
        self.collection_running = False
        self.log_file = log_file
        
    def log(self, message):
        """Log message to file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def save_game_data(self, data_dict):
        """Save game data if it's new"""
        if not data_dict:
            return False
            
        period = str(data_dict.get('period', ''))
        
        # Skip if already processed this period
        if period and self.last_processed_period == period:
            return False
            
        # Update last processed period
        self.last_processed_period = period
        
        # Save game data using data manager or directly to Supabase
        try:
            if USE_SUPABASE and "supabase_insert_table_data" in dir(builtins):
                try:
                    # Save directly to Supabase
                    result = supabase_insert_table_data(GAME_DATA_TABLE, data_dict)
                    
                    if not result.get('success', False):
                        # Fall back to data manager
                        add_new_game_data(
                            period=period,
                            price=float(data_dict.get('price', 0)),
                            number=str(data_dict.get('number', '')),
                            color=str(data_dict.get('color', ''))
                        )
                except Exception:
                    # Fall back to data manager
                    add_new_game_data(
                        period=period,
                        price=float(data_dict.get('price', 0)),
                        number=str(data_dict.get('number', '')),
                        color=str(data_dict.get('color', ''))
                    )
            else:
                # Use regular data manager
                add_new_game_data(
                    period=period,
                    price=float(data_dict.get('price', 0)),
                    number=str(data_dict.get('number', '')),
                    color=str(data_dict.get('color', ''))
                )
            self.log(f"Auto-saved game data for period {period}")
            return True
        except Exception as e:
            self.log(f"Error saving game data: {str(e)}")
            return False
    
    def start_auto_collection(self, df, interval=60):
        """Start background thread for automatic collection"""
        if self.collection_running:
            return
            
        self.collection_running = True
        self.log("Starting automatic data collection")
        
        # Get the latest period from the dataframe
        if len(df) > 0:
            self.last_processed_period = str(df.iloc[0]['period'])
            self.log(f"Initialized with latest period: {self.last_processed_period}")
        
        # No need to actually start a thread for our purposes

# Functions for data loading and visualization
def load_data(file_path='data/wingo_parity_data.csv'):
    """Load and prepare data from Supabase, with CSV as fallback"""
    
    # Determine which table to query based on file path
    table_name = GAME_DATA_TABLE if 'wingo_parity_data' in file_path else PREDICTION_DATA_TABLE
    
    # Always try loading from Supabase first
    if USE_SUPABASE and "supabase_fetch_table_data" in dir(builtins):
        try:
            # Query Supabase using REST API helper
            st.info(f"Fetching data from Supabase '{table_name}' table...")
            df = supabase_fetch_table_data(table_name)
            
            # Check if we got data
            if not df.empty:
                if 'timestamp' in df.columns:
                    try:
                        # Handle various timestamp formats from Supabase
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        
                        # Check if there are any NaT values after conversion
                        if df['timestamp'].isna().any():
                            st.warning(f"Some timestamp values could not be converted to datetime format. {df['timestamp'].isna().sum()} rows affected.")
                        
                        # Print the first few timestamp values for debugging
                        st.info(f"Sample timestamps: {df['timestamp'].iloc[:5].tolist()}")
                    except Exception as e:
                        st.warning(f"Could not convert timestamp column to datetime format: {str(e)}")
                        # Attempt a more aggressive conversion
                        try:
                            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
                        except Exception:
                            st.error("Failed to convert timestamps even with fallback method.")
                
                # Ensure period is consistently treated as string
                if 'period' in df.columns:
                    df['period'] = df['period'].astype(str)
                    
                st.success(f"Loaded {len(df)} records from Supabase '{table_name}' table")
                return df
            else:
                st.error(f"No data found in Supabase '{table_name}' table. Please make sure the table is populated.")
        except Exception as e:
            st.error(f"Error loading data from Supabase: {str(e)}")
    else:
        if not SUPABASE_READY:
            st.error(f"No Supabase connection available.")
    
    # Only as last resort, try loading from local CSV
    st.warning("Attempting to load from local CSV as fallback. Note that this is not the preferred data source.")
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            st.info(f"Loaded {len(df)} records from CSV file.")
            return df
        else:
            st.error(f"File not found: {file_path}")
            return pd.DataFrame(columns=['period', 'price', 'number', 'color', 'timestamp'])
            # Fallback method with minimal processing
            df = pd.read_csv(
                file_path, 
                dtype=object,  # Read everything as strings initially
                engine='python'  # More tolerant but slower engine
            )
        
        # Process the dataframe
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception:
                st.warning("Could not convert timestamp column to datetime format")
        
        # Ensure period is consistently treated as string
        if 'period' in df.columns:
            df['period'] = df['period'].astype(str)
            
        st.warning("Using local CSV data. Consider migrating this data to Supabase.")
        return df
    except Exception as e:
        st.error(f"Error loading data from CSV: {str(e)}")
        return None

def format_predictions(predictions_df):
    """Format predictions for display with enhanced confidence information"""
    # Create a copy to avoid modifying the original
    if predictions_df is None or len(predictions_df) == 0:
        return None
    
    df = predictions_df.copy()
    
    # Format confidence for display
    if 'confidence_numeric' in df.columns and len(df) > 0:
        # Ensure confidence is formatted as percentage with one decimal place
        df['confidence'] = df['confidence_numeric'].apply(lambda x: f"{x:.1f}%")
    elif 'confidence' in df.columns and len(df) > 0:
        # Check if confidence is already formatted
        sample_value = df['confidence'].iloc[0]
        if not isinstance(sample_value, str):
            df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1f}%")
            
    # Add visual confidence indicator if recommendation exists
    if 'recommendation' in df.columns:
        # Create emoji indicators for confidence levels
        def get_confidence_emoji(recommendation):
            if recommendation == "Strong":
                return "🟢 Strong"
            elif recommendation == "Moderate":
                return "🟡 Moderate"
            else:
                return "🔴 Weak"
        
        df['confidence_level'] = df['recommendation'].apply(get_confidence_emoji)
        
        # Reorder columns for better display
        columns_order = ['period']
        
        # Add target column (color or number)
        if 'color' in df.columns:
            columns_order.append('color')
        if 'number' in df.columns:
            columns_order.append('number')
            
        # Add confidence columns
        columns_order.extend(['confidence', 'confidence_level'])
        
        # Add any remaining columns
        for col in df.columns:
            if col not in columns_order and col not in ['confidence_numeric', 'recommendation', 'meets_threshold']:
                columns_order.append(col)
                
        # Select only the columns that exist in the dataframe
        existing_columns = [col for col in columns_order if col in df.columns]
        df = df[existing_columns]
    
    # Drop internal columns used for processing
    if 'confidence_numeric' in df.columns:
        df = df.drop('confidence_numeric', axis=1)
    if 'recommendation' in df.columns and 'confidence_level' in df.columns:
        df = df.drop('recommendation', axis=1)
    
    return df

def safe_dataframe_display(df, correct_col=None):
    """Ensures dataframe is properly typed for Streamlit display to avoid Arrow serialization issues"""
    # Create a copy to avoid modifying original
    display_df = df.copy()
    
    # If there's a correct column that needs to be converted to checkmarks
    if correct_col is not None and correct_col in display_df:
        display_df[correct_col] = display_df[correct_col].apply(lambda x: "✓" if x else "✗")
    
    # Convert all columns except boolean to string type to avoid type inference issues
    for col in display_df.columns:
        if col != correct_col:  # Skip the column we just formatted with checkmarks
            # Convert to string to avoid Arrow serialization issues
            display_df[col] = display_df[col].astype(str)
    
    return display_df

def display_pattern_analysis(patterns):
    """Display pattern analysis in a structured way"""
    st.subheader("Pattern Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'color_distribution' in patterns['patterns']:
            st.write("**Color Distribution**")
            try:
                # Convert to simple dataframe with string index
                color_dist_data = patterns['patterns']['color_distribution']
                # Limit to 10 items to avoid recursion issues
                if len(color_dist_data) > 10:
                    color_dist_data = {k: color_dist_data[k] for k in list(color_dist_data.keys())[:10]}
                
                color_dist = pd.DataFrame(list(color_dist_data.items()), columns=['Color', 'Count'])
                
                # Display as table
                st.write(color_dist)
            except Exception as e:
                st.warning(f"Could not display color distribution chart: {str(e)}")
                st.write(patterns['patterns']['color_distribution'])
            
        if 'color_max_streaks' in patterns['patterns']:
            st.write("**Maximum Color Streaks**")
            try:
                # Safely process data
                streak_data = patterns['patterns']['color_max_streaks']
                if len(streak_data) > 10:
                    streak_data = {k: streak_data[k] for k in list(streak_data.keys())[:10]}
                st.write(streak_data)
            except Exception as e:
                st.warning(f"Could not display streaks: {str(e)}")
    
    with col2:
        if 'number_distribution' in patterns['patterns']:
            st.write("**Number Distribution**")
            try:
                # Convert to simple dataframe with string index
                num_dist_data = patterns['patterns']['number_distribution']
                if len(num_dist_data) > 10:
                    num_dist_data = {k: num_dist_data[k] for k in list(num_dist_data.keys())[:10]}
                    
                num_df = pd.DataFrame(list(num_dist_data.items()), columns=['Number', 'Count'])
                
                # Display as table
                st.write(num_df)
            except Exception as e:
                st.warning(f"Could not display number distribution chart: {str(e)}")
                st.write(patterns['patterns']['number_distribution'])
            
        if 'even_odd' in patterns['patterns']:
            st.write("**Even vs Odd Distribution**")
            try:
                # Safely process data
                even_odd_data = patterns['patterns']['even_odd']
                st.write(even_odd_data)
            except Exception as e:
                st.warning(f"Could not display even/odd distribution: {str(e)}")
    
    # Display transitions
    if 'color_transitions' in patterns['patterns']:
        st.write("**Color Transition Patterns (Top 5)**")
        try:
            trans_data = patterns['patterns']['color_transitions']
            # Only take top 5 transitions
            if len(trans_data) > 5:
                top_5_trans = dict(sorted(trans_data.items(), key=lambda x: x[1], reverse=True)[:5])
                st.write(top_5_trans)
            else:
                st.write(trans_data)
        except Exception as e:
            st.warning(f"Could not display transitions: {str(e)}")

try:
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize the data collectors if not already done
    if 'data_collector' not in st.session_state or st.session_state.data_collector is None:
        st.session_state.data_collector = AutoDataCollector()
        st.session_state.last_update_time = datetime.datetime.now()

    # Load data from Supabase
    if USE_SUPABASE and SUPABASE_READY:
        # Load game data
        df = load_game_data_from_supabase()
        if not df.empty:
            print(f"Loaded {len(df)} records from Supabase")
            st.sidebar.success(f"✅ Loaded {len(df)} records from Supabase '{GAME_DATA_TABLE}' table")
        else:
            print("No data found in Supabase table")
            # If Supabase returned empty dataframe, create a minimal structure
            df = pd.DataFrame(columns=['period', 'price', 'number', 'color', 'timestamp'])
    else:
        # Only show this error if we're supposed to use Supabase but it's not ready
        if USE_SUPABASE and not SUPABASE_READY:
            st.error("No Supabase connection available.")
            st.warning("Attempting to load from local CSV as fallback. Note that this is not the preferred data source.")
            # Try loading from local CSV as fallback
            if os.path.exists(GAME_DATA_CSV):
                try:
                    df = pd.read_csv(GAME_DATA_CSV)
                    st.info(f"Loaded {len(df)} records from local CSV file.")
                except Exception as e:
                    print(f"Error loading CSV: {e}")
                    df = pd.DataFrame(columns=['period', 'price', 'number', 'color', 'timestamp'])
            else:
                df = pd.DataFrame(columns=['period', 'price', 'number', 'color', 'timestamp'])
        else:
            df = pd.DataFrame(columns=['period', 'price', 'number', 'color', 'timestamp'])

    # Initialize the predictions CSV file with headers if it doesn't exist
    predictions_csv_path = 'data/predictions_data.csv'
    if not os.path.exists(predictions_csv_path):
        with open(predictions_csv_path, 'w') as f:
            header = "period,timestamp,predicted_number,predicted_color,number_confidence,color_confidence,"
            header += "number_recommendation,color_recommendation,actual_number,actual_color,correct_number,correct_color\n"
            f.write(header)

    # Initialize prediction objects if they don't exist
    if 'predictor_color' not in st.session_state:
        st.session_state.predictor_color = GamePredictor(model_dir="models", target_type="color")
    if 'predictor_number' not in st.session_state:
        st.session_state.predictor_number = GamePredictor(model_dir="models", target_type="number")
        
    # Initialize other session state variables if they don't exist
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'color_predictions' not in st.session_state:
        st.session_state.color_predictions = None
    if 'number_predictions' not in st.session_state:
        st.session_state.number_predictions = None
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 50.0
    if 'prediction_periods' not in st.session_state:
        st.session_state.prediction_periods = 5
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True  # Always enable auto-refresh by default
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 40
    if 'show_raw_data' not in st.session_state:
        st.session_state.show_raw_data = False
    if 'display_settings' not in st.session_state:
        st.session_state.display_settings = {}
    if 'hide_accuracy' not in st.session_state:
        st.session_state.hide_accuracy = False
    if 'auto_save_data' not in st.session_state:
        st.session_state.auto_save_data = True  # Enable auto-save by default
    
    # Tracking variables
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = None
    if 'accuracy_history' not in st.session_state:
        st.session_state.accuracy_history = {'color': [], 'number': []}
    if 'predicted_history' not in st.session_state:
        st.session_state.predicted_history = []
    if 'actual_results' not in st.session_state:
        st.session_state.actual_results = {}
        
except Exception as e:
    st.error(f"Error initializing session state: {str(e)}")
    # Force initialization of critical variables
    st.session_state.predictor_color = GamePredictor(model_dir="models", target_type="color")
    st.session_state.predictor_number = GamePredictor(model_dir="models", target_type="number")
    st.session_state.models_trained = False
    st.session_state.color_predictions = None
    st.session_state.number_predictions = None
    st.session_state.confidence_threshold = 50.0
    st.session_state.prediction_periods = 5
    st.session_state.auto_refresh = False
    st.session_state.refresh_interval = 40
    st.session_state.show_raw_data = False
    st.session_state.auto_save_data = True  # Enable auto-save by default
    st.session_state.last_update_time = None
    st.session_state.accuracy_history = {'color': [], 'number': []}
    st.session_state.predicted_history = []
    st.session_state.actual_results = {}

def save_prediction_to_supabase(prediction_data):
    """Save prediction data to Supabase predictions_data table"""
    try:
        if supabase_client is None:
            print("No Supabase client available for saving prediction data")
            return False
            
        # Insert data into Supabase
        response = supabase_client.table(PREDICTION_DATA_TABLE).insert(prediction_data).execute()
        print(f"Saved prediction to Supabase: {response}")
        st.success("✅ Prediction saved to Supabase cloud database!")
        return True
    except Exception as e:
        print(f"Error saving prediction to Supabase: {str(e)}")
        st.error(f"Error saving prediction to Supabase: {str(e)}")
        return False

def init_models(df, force_train=False):
    """Update predictions with latest data with confidence threshold filtering"""
    # Initialize confidence threshold if not already set
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 50.0
    
    # Check if df is empty or None before proceeding
    if df is None or df.empty:
        print("Warning: Empty dataframe provided to init_models. Cannot train models or make predictions.")
        st.warning("No data available for model training and prediction. Please add game data first.")
        # Initialize with empty predictions
        st.session_state.color_predictions = pd.DataFrame()
        st.session_state.number_predictions = pd.DataFrame()
        st.session_state.models_trained = False
        return
    
    # Make sure models are trained first
    if force_train or not st.session_state.models_trained:
        with st.spinner('Training models...'):
            # Train color predictor
            st.session_state.predictor_color.train_models(df)
            
            # Train number predictor
            st.session_state.predictor_number.train_models(df)
            
            # Update training flag
            st.session_state.models_trained = True
    
    # Make color predictions
    color_predictions = st.session_state.predictor_color.predict_next_periods(
        df, periods=st.session_state.prediction_periods, use_best_model=True
    )
    
    # Make number predictions
    number_predictions = st.session_state.predictor_number.predict_next_periods(
        df, periods=st.session_state.prediction_periods, use_best_model=True
    )
    
    # Update last update time
    st.session_state.last_update_time = datetime.datetime.now()
    
    # Apply confidence thresholding to color predictions
    if color_predictions is not None:
        # Convert confidence to numeric if it's not already
        if 'confidence' in color_predictions.columns:
            if color_predictions['confidence'].dtype == object:
                try:
                    color_predictions['confidence_numeric'] = color_predictions['confidence'].str.rstrip('%').astype(float)
                except:
                    color_predictions['confidence_numeric'] = color_predictions['confidence']
            else:
                color_predictions['confidence_numeric'] = color_predictions['confidence']
                
            # Filter by confidence threshold
            filtered_color = color_predictions[color_predictions['confidence_numeric'] >= st.session_state.confidence_threshold].copy()
            # Add a confidence indicator
            filtered_color['recommendation'] = filtered_color['confidence_numeric'].apply(
                lambda x: "Strong" if x >= 75 else "Moderate" if x >= 50 else "Weak"
            )
            st.session_state.color_predictions = filtered_color
        else:
            st.session_state.color_predictions = color_predictions
    
    # Apply confidence thresholding to number predictions
    if number_predictions is not None:
        # Convert confidence to numeric if it's not already
        if 'confidence' in number_predictions.columns:
            if number_predictions['confidence'].dtype == object:
                try:
                    number_predictions['confidence_numeric'] = number_predictions['confidence'].str.rstrip('%').astype(float)
                except:
                    number_predictions['confidence_numeric'] = number_predictions['confidence']
            else:
                number_predictions['confidence_numeric'] = number_predictions['confidence']
                
            # Filter by confidence threshold
            filtered_number = number_predictions[number_predictions['confidence_numeric'] >= st.session_state.confidence_threshold].copy()
            # Add a confidence indicator
            filtered_number['recommendation'] = filtered_number['confidence_numeric'].apply(
                lambda x: "Strong" if x >= 75 else "Moderate" if x >= 50 else "Weak"
            )
            st.session_state.number_predictions = filtered_number
        else:
            st.session_state.number_predictions = number_predictions
    
    # Store predicted periods for accuracy tracking
    if st.session_state.color_predictions is not None:
        for _, row in st.session_state.color_predictions.iterrows():
            period = row['period']
            st.session_state.predicted_history.append({
                'period': period,
                'predicted_color': row['color'],
                'predicted_number': None,
                'confidence_color': row.get('confidence_numeric', 0),
                'predicted_time': st.session_state.last_update_time
            })
        
    if st.session_state.number_predictions is not None:
        for _, row in st.session_state.number_predictions.iterrows():
            period = row['period']
            # Find existing prediction
            for pred in st.session_state.predicted_history:
                if pred['period'] == period:
                    pred['predicted_number'] = row['number']
                    pred['confidence_number'] = row.get('confidence_numeric', 0)
                    break

def update_predictions(predictor_color, predictor_number):
    """Update prediction data using the trained models"""
    global df
    
    if not hasattr(st.session_state, 'predictor_color') or st.session_state.predictor_color is None:
        st.warning("Color predictor not available. Try retraining the models.")
        return
    
    if not hasattr(st.session_state, 'predictor_number') or st.session_state.predictor_number is None:
        st.warning("Number predictor not available. Try retraining the models.")
        return
    
    # Get predictions for the next periods
    if 'prediction_periods' in st.session_state:
        periods = st.session_state.prediction_periods
    else:
        periods = 5  # Default
    
    # Get confidence threshold from session state or use default
    confidence_threshold = st.session_state.confidence_threshold if hasattr(st.session_state, 'confidence_threshold') else 50.0
    
    # Get color predictions with confidence threshold applied
    color_predictions = predictor_color.predict_next_periods(
        df, 
        periods=periods, 
        confidence_threshold=confidence_threshold
    )
    
    # Get number predictions with confidence threshold applied
    number_predictions = predictor_number.predict_next_periods(
        df, 
        periods=periods, 
        confidence_threshold=confidence_threshold
    )
    
    # Store full predictions in session state
    st.session_state.color_predictions_full = color_predictions
    st.session_state.number_predictions_full = number_predictions
    
    # Apply confidence threshold filtering if predictions exist
    if len(color_predictions) > 0:
        # Filter predictions that meet the threshold
        color_predictions_filtered = color_predictions[color_predictions['meets_threshold'] == True]
        st.session_state.color_predictions = color_predictions_filtered
    else:
        st.session_state.color_predictions = pd.DataFrame()
        
    if len(number_predictions) > 0:
        # Filter predictions that meet the threshold
        number_predictions_filtered = number_predictions[number_predictions['meets_threshold'] == True]
        st.session_state.number_predictions = number_predictions_filtered
    else:
        st.session_state.number_predictions = pd.DataFrame()
    
    # Get current time for tracking prediction
    now = datetime.datetime.now()
    st.session_state.last_update_time = now
    
    # Store the first prediction in history regardless of confidence
    if len(color_predictions) > 0 and len(number_predictions) > 0:
        next_period = color_predictions.iloc[0]['period']
        next_color = color_predictions.iloc[0]['color']
        next_color_confidence = color_predictions.iloc[0].get('confidence', 'Unknown')
        next_color_recommendation = color_predictions.iloc[0].get('recommendation', 'Unknown')
        next_number = number_predictions.iloc[0]['number']
        next_number_confidence = number_predictions.iloc[0].get('confidence', 'Unknown')
        next_number_recommendation = number_predictions.iloc[0].get('recommendation', 'Unknown')
        
        # Store in prediction history for tracking
        prediction_data = {
            'period': next_period,
            'predicted_color': next_color,
            'color_confidence': next_color_confidence,
            'color_recommendation': next_color_recommendation,
            'predicted_number': next_number,
            'number_confidence': next_number_confidence,
            'number_recommendation': next_number_recommendation,
            'predicted_time': now
        }
        
        st.session_state.predicted_history.append(prediction_data)
        
        # Always save prediction data to separate CSV file
        try:
            # Check if we already have actual results for this period in the dataframe
            actual_number = None
            actual_color = None
            correct_number = None
            correct_color = None
            
            # Convert periods to string for safe comparison
            if df is not None and len(df) > 0:
                df_periods = df['period'].astype(str).tolist()
                
                # If we have an actual result for this period
                if next_period in df_periods:
                    # Get actual values from dataframe
                    index = df_periods.index(next_period)
                    row = df.iloc[index]
                    actual_number = row['number']
                    actual_color = row['color']
                    
                    # Calculate if prediction was correct
                    correct_number = str(actual_number == next_number).lower()
                    correct_color = str(actual_color == next_color).lower()
            
            # Save prediction data with actual results when available
            save_prediction_data(
                period=next_period,
                predicted_number=next_number,
                predicted_color=next_color,
                number_confidence=next_number_confidence,
                color_confidence=next_color_confidence,
                number_recommendation=next_number_recommendation,
                color_recommendation=next_color_recommendation,
                actual_number=actual_number,
                actual_color=actual_color,
                correct_number=correct_number,
                correct_color=correct_color,
                timestamp=now
            )
            st.success(f"Prediction for period {next_period} saved to {PREDICTION_DATA_CSV}")
        except Exception as e:
            st.warning(f"Could not save prediction data: {str(e)}")

def update_accuracy(df):
    """Update accuracy by comparing predictions with actual results"""
    # Convert periods to string for safe comparison
    df_periods = df['period'].astype(str).tolist()
    
    # Check if there are new periods with actual data that have predictions
    for pred in st.session_state.predicted_history:
        period = str(pred['period'])  # Ensure period is string
        
        # If we have an actual result for this period
        if period in df_periods:
            # Get index of this period in the dataframe
            index = df_periods.index(period)
            row = df.iloc[index]
            
            # Get predicted and actual values
            predicted_color = pred.get('predicted_color')
            actual_color = row['color']
            predicted_number = pred.get('predicted_number')
            actual_number = row['number']
            
            # Skip if we've already evaluated this period
            if period in st.session_state.actual_results:
                continue
                
            # Mark this period as evaluated
            st.session_state.actual_results[period] = True
            
            # Update the predictions CSV with actual results
            try:
                # Check if color and number predictions were correct
                correct_color = predicted_color == actual_color if predicted_color is not None else None
                correct_number = str(predicted_number) == str(actual_number) if predicted_number is not None else None
                
                # Save to predictions CSV
                save_prediction_data(
                    period=period,
                    predicted_number=predicted_number,
                    predicted_color=predicted_color,
                    number_confidence=pred.get('number_confidence', None),
                    color_confidence=pred.get('color_confidence', None),
                    number_recommendation=pred.get('number_recommendation', None),
                    color_recommendation=pred.get('color_recommendation', None),
                    actual_number=actual_number,
                    actual_color=actual_color,
                    correct_number=str(correct_number).lower() if correct_number is not None else None,
                    correct_color=str(correct_color).lower() if correct_color is not None else None,
                    timestamp=pred.get('predicted_time', datetime.datetime.now())
                )
            except Exception as e:
                st.warning(f"Could not update prediction data with actual results: {str(e)}")
            
            # Check if color prediction was correct
            if predicted_color is not None and actual_color is not None:
                is_correct = predicted_color == actual_color
                st.session_state.accuracy_history['color'].append({
                    'period': str(period),
                    'predicted': str(predicted_color),
                    'actual': str(actual_color),
                    'correct': bool(is_correct),  # Use boolean for boolean values
                    'confidence': str(pred.get('color_confidence', 'Unknown')),  # Add confidence percentage as string
                    'confidence_level': str(pred.get('color_recommendation', 'Unknown')),  # Add recommendation level
                    'timestamp': pred['predicted_time']
                })
            
            # Check if number prediction was correct
            if predicted_number is not None and actual_number is not None:
                is_correct = str(predicted_number) == str(actual_number)
                st.session_state.accuracy_history['number'].append({
                    'period': str(period),
                    'predicted': str(predicted_number),
                    'actual': str(actual_number),
                    'correct': bool(is_correct),  # Use boolean for boolean values
                    'confidence': str(pred.get('number_confidence', 'Unknown')),  # Add confidence percentage as string
                    'confidence_level': str(pred.get('number_recommendation', 'Unknown')),  # Add recommendation level
                    'timestamp': pred['predicted_time']
                })

# Define helper functions that might be missing from builtins

# Function to insert data into Supabase table
def supabase_insert_table_data(table_name, data_dict):
    """Insert data into Supabase table"""
    if not hasattr(builtins, 'supabase_insert_table_data'):
        print("No supabase_insert_table_data function found in builtins")
        try:
            if supabase_client is not None:
                # Check if this is a prediction and has a period
                if table_name == PREDICTION_DATA_TABLE and 'period' in data_dict:
                    # Check if a record with this period already exists
                    period = data_dict['period']
                    existing_response = supabase_client.table(table_name).select("id").eq("period", str(period)).execute()
                    
                    existing_data = []
                    if hasattr(existing_response, 'data'):
                        existing_data = existing_response.data
                    elif isinstance(existing_response, dict) and 'data' in existing_response:
                        existing_data = existing_response['data']
                        
                    if existing_data:
                        # Record exists, update it
                        record_id = existing_data[0]['id']
                        print(f"Found existing record for period {period}. Updating instead of inserting.")
                        response = supabase_client.table(table_name).update(data_dict).eq("id", record_id).execute()
                    else:
                        # No record exists, insert new one
                        print(f"No existing record found for period {period}. Inserting new record.")
                        response = supabase_client.table(table_name).insert(data_dict).execute()
                else:
                    # For other tables or data without period, just insert
                    response = supabase_client.table(table_name).insert(data_dict).execute()
                
                if hasattr(response, 'data') and response.data:
                    print(f"Successfully inserted/updated data in {table_name}")
                    return True
                else:
                    print(f"Failed to insert/update data in Supabase table {table_name}")
                    return False
            else:
                print(f"No Supabase client available, cannot insert data into {table_name}")
                return False
        except Exception as e:
            print(f"Error inserting data into Supabase: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
    else:
        return builtins.supabase_insert_table_data(table_name, data_dict)

# Function to fetch data from Supabase table
def supabase_fetch_table_data(table_name):
    """Fetch data from Supabase table"""
    if not hasattr(builtins, 'supabase_fetch_table_data'):
        print("No supabase_fetch_table_data function found in builtins")
        try:
            if supabase_client is not None:
                # First, get the total count of records
                count_response = supabase_client.table(table_name).select("*", count='exact').limit(1).execute()
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
                    response = supabase_client.table(table_name).select("*").range(offset, offset + batch_size - 1).execute()
                    
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
                
                # Convert to DataFrame
                df = pd.DataFrame(all_data)
                
                # Process timestamps if they exist
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching data from Supabase: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
    else:
        return builtins.supabase_fetch_table_data(table_name)

# Function to add game data to CSV file
def add_game_data_to_csv(data_dict, file_path):
    """Add game data to CSV file"""
    try:
        # Check if file exists
        if os.path.exists(file_path):
            # Read existing data
            try:
                existing_df = pd.read_csv(
                    file_path,
                    dtype={'period': str, 'number': str, 'color': str, 'price': float}
                )
            except Exception:
                # Fallback with minimal processing
                existing_df = pd.read_csv(file_path, dtype=object)
                
            # Create new dataframe with the new data
            new_entry = pd.DataFrame([data_dict])
            
            # Combine new entry with existing data (new data at the top)
            updated_df = pd.concat([new_entry, existing_df], ignore_index=True)
        else:
            # Create new file with just this entry
            updated_df = pd.DataFrame([data_dict])
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the updated dataframe back to CSV
        updated_df.to_csv(file_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving data to CSV: {str(e)}")
        return False

# Load prediction history from Supabase to initialize the session state prediction history
def load_prediction_history():
    """Load prediction history from Supabase to initialize the session state prediction history"""
    try:
        # Only proceed if Supabase is available
        if not (USE_SUPABASE and SUPABASE_READY and supabase_client is not None):
            print("Supabase not available for loading prediction history")
            return
            
        print("Loading prediction history from Supabase...")
        predictions_df = load_prediction_data_from_supabase()
        
        if predictions_df is None or predictions_df.empty:
            print("No prediction history found in Supabase")
            return
            
        print(f"Loaded {len(predictions_df)} prediction records from Supabase")
        
        # Initialize prediction history
        prediction_history = []
        
        # Convert timestamps to datetime objects
        if 'timestamp' in predictions_df.columns:
            predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'], errors='coerce')
        
        # Sort by period to ensure chronological order
        if 'period' in predictions_df.columns:
            predictions_df = predictions_df.sort_values(by='period')
        
        # Populate prediction history
        for _, row in predictions_df.iterrows():
            prediction_data = {
                'period': row.get('period', ''),
                'predicted_color': row.get('predicted_color', ''),
                'color_confidence': row.get('color_confidence', ''),
                'color_recommendation': row.get('color_recommendation', ''),
                'predicted_number': row.get('predicted_number', ''),
                'number_confidence': row.get('number_confidence', ''),
                'number_recommendation': row.get('number_recommendation', ''),
                'predicted_time': row.get('timestamp', datetime.datetime.now())
            }
            prediction_history.append(prediction_data)
            
            # Update actual results dictionary to mark this prediction as evaluated if results exist
            if 'actual_number' in row and 'actual_color' in row and not pd.isna(row['actual_number']) and not pd.isna(row['actual_color']):
                if 'actual_results' in st.session_state:
                    st.session_state.actual_results[str(row['period'])] = True
        
        # Update session state with loaded history
        if prediction_history:
            st.session_state.predicted_history = prediction_history
            print(f"Updated prediction history with {len(prediction_history)} records")
        
    except Exception as e:
        print(f"Error loading prediction history: {str(e)}")
        import traceback
        print(traceback.format_exc())

# Function to get overall prediction statistics from Supabase
def get_overall_prediction_stats():
    """Get overall prediction statistics from Supabase database"""
    try:
        # Only proceed if Supabase is available
        if not (USE_SUPABASE and SUPABASE_READY and supabase_client is not None):
            print("Supabase not available for fetching prediction stats")
            return None
            
        print("Fetching overall prediction statistics from Supabase...")
        predictions_df = load_prediction_data_from_supabase()
        
        if predictions_df is None or predictions_df.empty:
            print("No prediction data found in Supabase")
            return None
        
        total_predictions = len(predictions_df)
        
        # Initialize counters
        color_stats = {
            'total': 0,
            'correct': 0,
            'accuracy': 0
        }
        
        number_stats = {
            'total': 0,
            'correct': 0,
            'accuracy': 0
        }
        
        # Count predictions with actual results
        predictions_with_results = 0
        
        # Process the data
        for _, row in predictions_df.iterrows():
            # Check if this prediction has actual results
            has_actual_results = (
                'actual_color' in row and not pd.isna(row['actual_color']) and 
                'actual_number' in row and not pd.isna(row['actual_number'])
            )
            
            if has_actual_results:
                predictions_with_results += 1
                
                # Process color predictions
                if 'correct_color' in row and not pd.isna(row['correct_color']):
                    color_stats['total'] += 1
                    # Convert to boolean if it's a string
                    if isinstance(row['correct_color'], str):
                        is_correct = row['correct_color'].lower() == 'true'
                    else:
                        is_correct = bool(row['correct_color'])
                        
                    if is_correct:
                        color_stats['correct'] += 1
                
                # Process number predictions
                if 'correct_number' in row and not pd.isna(row['correct_number']):
                    number_stats['total'] += 1
                    # Convert to boolean if it's a string
                    if isinstance(row['correct_number'], str):
                        is_correct = row['correct_number'].lower() == 'true'
                    else:
                        is_correct = bool(row['correct_number'])
                        
                    if is_correct:
                        number_stats['correct'] += 1
        
        # Calculate accuracy percentages
        if color_stats['total'] > 0:
            color_stats['accuracy'] = (color_stats['correct'] / color_stats['total']) * 100
            
        if number_stats['total'] > 0:
            number_stats['accuracy'] = (number_stats['correct'] / number_stats['total']) * 100
        
        return {
            'total_predictions': total_predictions,
            'predictions_with_results': predictions_with_results,
            'color_stats': color_stats,
            'number_stats': number_stats
        }
        
    except Exception as e:
        print(f"Error getting overall prediction stats: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

# Main dashboard
st.title("Game Prediction Dashboard")

with st.sidebar:
    st.header("Controls")
    data_file = st.text_input("Data file path", "data/wingo_parity_data.csv")
    
    st.subheader("Add New Game Data")
    with st.form("new_data_form"):
        period_input = st.text_input("Period ID", placeholder="e.g., 202506161230")
        price_input = st.text_input("Price", placeholder="e.g., 35000.00")
        number_input = st.number_input("Number", min_value=0, max_value=9, step=1)
        color_input = st.selectbox("Color", options=["Red", "Green", "Violet", "Red+Violet", "Green+Violet"])
        
        submit_button = st.form_submit_button("Add Data")
        
        if submit_button and period_input:
            if period_input and price_input:
                try:
                    new_data = {
                        "period": period_input,
                        "price": price_input,
                        "number": int(number_input),
                        "color": color_input,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    
                    # Add to Supabase if available
                    if USE_SUPABASE and SUPABASE_READY:
                        try:
                            response = supabase_client.table(GAME_DATA_TABLE).insert(new_data).execute()
                            print(f"Successfully added new data to {GAME_DATA_TABLE}: {response}")
                            st.success(f"Data added successfully to Supabase cloud database!")
                            time.sleep(1)  # Short pause to allow UI update
                            st.rerun()  # Rerun the app to refresh data
                        except Exception as e:
                            print(f"Error adding data to Supabase: {str(e)}")
                            st.error(f"Failed to add data to Supabase: {str(e)}")
                            # Fallback to CSV
                            add_game_data_to_csv(new_data, GAME_DATA_CSV)
                            st.warning("Data added to local CSV file as fallback.")
                    else:
                        # Save to CSV if Supabase is not available
                        add_game_data_to_csv(new_data, GAME_DATA_CSV)
                        st.success("Data added successfully to CSV file!")
                        time.sleep(1)  # Short pause to allow UI update
                        st.rerun()  # Rerun the app to refresh data
                except Exception as e:
                    st.error(f"Error adding data: {str(e)}")
            else:
                st.error("Please fill out all required fields.")
    
    st.subheader("Model Training")
    retrain_button = st.button("Retrain Models")
    
    st.subheader("Auto-Refresh Settings")
    # Always enable auto-refresh (no checkbox)
    st.session_state.auto_refresh = True
    # Just show informational text that auto-refresh is enabled
    st.info("✓ Auto-refresh is always enabled for real-time predictions")
    
    # Only allow configuring the refresh interval
    refresh_interval = st.slider("Refresh interval (seconds)", 
                               min_value=30, max_value=300, value=st.session_state.refresh_interval)
    
    # Update refresh interval in session state
    if refresh_interval != st.session_state.refresh_interval:
        st.session_state.refresh_interval = refresh_interval
    
    st.subheader("Prediction Settings")
    prediction_periods = st.slider("Number of periods to predict", 
                                min_value=1, max_value=10, value=st.session_state.prediction_periods)
                                
    # Add confidence threshold slider
    confidence_threshold = st.slider(
        "Minimum confidence threshold (%)",
        min_value=0.0,
        max_value=100.0,
        value=st.session_state.confidence_threshold,
        step=5.0,
        help="Only show predictions with confidence above this threshold"
    )
    
    # Update session state based on UI inputs
    if prediction_periods != st.session_state.prediction_periods:
        st.session_state.prediction_periods = prediction_periods
        # Don't update predictions yet, will do in main flow
    
    if confidence_threshold != st.session_state.confidence_threshold:
        st.session_state.confidence_threshold = confidence_threshold
        # Don't update predictions yet, will do in main flow
    
    st.subheader("Display Settings")
    show_raw_data = st.checkbox("Show raw data", value=st.session_state.show_raw_data)
    
    if show_raw_data != st.session_state.show_raw_data:
        st.session_state.show_raw_data = show_raw_data
    
    # Manual refresh button
    if st.button("Refresh Now"):
        st.rerun()

# We should already have df loaded from Supabase at this point
# If not, ensure we have a valid df variable
if 'df' not in locals() or df is None or df.empty:
    print("DataFrame not found in locals or empty, trying to load data explicitly")
    # Try to get data from Supabase again
    if USE_SUPABASE and SUPABASE_READY and supabase_client:
        df = load_game_data_from_supabase()
        print(f"Re-loaded {len(df) if df is not None else 0} records from Supabase")
        
        # Load prediction history from Supabase if it's empty
        if 'predicted_history' in st.session_state and len(st.session_state.predicted_history) == 0:
            load_prediction_history()
    else:
        # Fallback to local CSV as absolute last resort
        print("Falling back to local CSV as last resort")
        df = load_data(data_file)
        
# IMPORTANT: Ensure df is never None to avoid errors
if df is None:
    print("WARNING: df is None after all loading attempts")
    df = pd.DataFrame(columns=['period', 'price', 'number', 'color', 'timestamp'])

# Check if we need to force update predictions based on new data
if df is not None and len(df) > 0:
    if 'last_data_period' not in st.session_state:
        st.session_state.last_data_period = None
        
    # If we have new data (first period has changed), force predictions update
    current_period = str(df.iloc[0]['period'])
    if current_period != st.session_state.last_data_period:
        st.session_state.force_predict = True
        st.session_state.last_data_period = current_period
        st.info(f"New data detected for period {current_period}, generating predictions...")

if df is not None:
    # Initialize models
    init_models(df, force_train=retrain_button)
    
    # Initialize force_predict flag if not exists
    if 'force_predict' not in st.session_state:
        st.session_state.force_predict = False
        
    # Check if we need to update predictions
    current_time = datetime.datetime.now()
    try:
        time_diff = (current_time - st.session_state.last_update_time).total_seconds()
        needs_refresh = time_diff > refresh_interval
    except Exception as e:
        print(f"Error calculating refresh time: {str(e)}")
        # Reset last_update_time to ensure it's a datetime
        st.session_state.last_update_time = current_time
        needs_refresh = True
        
    if (st.session_state.force_predict or
        st.session_state.last_update_time is None or needs_refresh):
        update_predictions(st.session_state.predictor_color, st.session_state.predictor_number)
        # Reset force flag after update
        st.session_state.force_predict = False
    
    # Update accuracy with latest data
    update_accuracy(df)
    
    # Main dashboard display
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Color Predictions")
        if st.session_state.color_predictions is not None and len(st.session_state.color_predictions) > 0:
            # Format and display predictions
            formatted_df = format_predictions(st.session_state.color_predictions)
            if formatted_df is not None:
                # Create custom columns for cleaner display
                columns = []
                for i, row in formatted_df.iterrows():
                    period = row['period']
                    color = row['color']
                    confidence = row['confidence'] if 'confidence' in row else "-"
                    confidence_level = row['confidence_level'] if 'confidence_level' in row else "-"
                    
                    col = st.columns(4)
                    with col[0]:
                        st.write(f"**{period}**")
                    with col[1]:
                        st.write(f"**{color}**")
                    with col[2]:
                        st.write(f"**{confidence}**")
                    with col[3]:
                        st.write(f"{confidence_level}")
                    
                    columns.append(col)
            
            # Show history and prediction visualization if data is available
            try:
                if len(df) > 0:
                    fig = st.session_state.predictor_color.plot_prediction_history(
                        df.head(10), 
                        st.session_state.color_predictions
                    )
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not display color history: {str(e)}")
        else:
            st.info("No color predictions available. Try retraining the models.")
            try:
                # Display recent history
                last_10_records = df.head(10).copy()
                if 'color' in last_10_records.columns:
                    st.write("**Recent Color History:**")
                    color_history = last_10_records[['period', 'color']].copy()
                    st.dataframe(color_history)
            except Exception as e:
                st.warning(f"Could not display color history: {str(e)}")
    
    with col2:
        st.header("Number Predictions")
        if st.session_state.number_predictions is not None and len(st.session_state.number_predictions) > 0:
            # Format and display predictions
            formatted_df = format_predictions(st.session_state.number_predictions)
            if formatted_df is not None:
                # Create column headers first
                cols = st.columns(4)
                with cols[0]:
                    st.write("**period**")
                with cols[1]:
                    st.write("**number**")
                with cols[2]:
                    st.write("**confidence**")
                with cols[3]:
                    st.write("**confidence level**")
                
                # Create custom columns for cleaner display
                columns = []
                for i, row in formatted_df.iterrows():
                    period = row['period']
                    number = row['number']
                    confidence = row['confidence'] if 'confidence' in row else "-"
                    confidence_level = row['confidence_level'] if 'confidence_level' in row else "-"
                    
                    col = st.columns(4)
                    with col[0]:
                        st.write(f"{period}")
                    with col[1]:
                        st.write(f"{number}")
                    with col[2]:
                        st.write(f"{confidence}")
                    with col[3]:
                        st.write(f"{confidence_level}")
                    
                    columns.append(col)
                
            # Show history and prediction visualization if data is available
            try:
                if len(df) > 0:
                    fig = st.session_state.predictor_number.plot_prediction_history(
                        df.head(10), 
                        st.session_state.number_predictions
                    )
                    st.pyplot(fig)
            except Exception as e:
                st.warning(f"Could not display number history: {str(e)}")
        else:
            st.info("No number predictions available. Try retraining the models.")
            try:
                # Display recent history
                last_10_records = df.head(10).copy()
                if 'number' in last_10_records.columns:
                    st.write("**Recent Number History:**")
                    number_history = last_10_records[['period', 'number']].copy()
                    st.dataframe(number_history)
            except Exception as e:
                st.warning(f"Could not display number history: {str(e)}")
    
    # Display pattern analysis
    patterns = st.session_state.predictor_color.analyze_patterns(df, lookback=30)
    display_pattern_analysis(patterns)
    
    # Show prediction accuracy
    st.header("Prediction Accuracy")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Color Prediction Accuracy")
        if st.session_state.accuracy_history['color']:
            # Convert to DataFrame with consistent types
            accuracy_df = pd.DataFrame(st.session_state.accuracy_history['color'])
            
            # Ensure correct column is boolean type for proper calculation
            accuracy_df['correct'] = accuracy_df['correct'].astype(bool)
            
            # Calculate accuracy percentage
            correct_count = accuracy_df['correct'].sum()
            total_count = len(accuracy_df)
            correct_pct = (correct_count / total_count * 100) if total_count > 0 else 0
            
            # Calculate accuracy by confidence level if confidence column exists
            if 'confidence' in accuracy_df.columns:
                st.metric("Overall Accuracy", f"{correct_pct:.1f}% ({correct_count}/{total_count} correct)")
                
                # Group by confidence level to show accuracy for each level
                confidence_metrics = accuracy_df.groupby('confidence')['correct'].agg(['mean', 'count', 'sum']).reset_index()
                confidence_metrics['accuracy'] = confidence_metrics['mean'] * 100
                
                # Display accuracy by confidence level
                st.subheader("Accuracy by Confidence Level")
                for _, row in confidence_metrics.iterrows():
                    conf_level = row['confidence']
                    conf_acc = row['accuracy']
                    conf_count = row['count']
                    true_count = row['sum']
                    st.metric(
                        f"{conf_level} Predictions ({conf_count})", 
                        f"{conf_acc:.1f}% ({true_count}/{conf_count} correct)"
                    )
            else:
                # Just display the overall metric if no confidence data
                st.metric("Accuracy", f"{correct_pct:.1f}%")
            
            # Reorder columns to show confidence prominently
            cols = ['period', 'predicted', 'actual', 'correct', 'confidence']
            display_cols = [col for col in cols if col in accuracy_df.columns]
            ordered_df = accuracy_df[display_cols]
            
            # Use our safe display function to avoid Arrow serialization issues
            display_df = safe_dataframe_display(ordered_df, correct_col='correct')
                
            st.dataframe(display_df)
    
    with col2:
        st.subheader("Number Prediction Accuracy")
        if st.session_state.accuracy_history['number']:
            # Convert to DataFrame with consistent types
            accuracy_df = pd.DataFrame(st.session_state.accuracy_history['number'])
            
            # Ensure correct column is boolean type for proper calculation
            accuracy_df['correct'] = accuracy_df['correct'].astype(bool)
            
            # Calculate accuracy percentage
            correct_count = accuracy_df['correct'].sum()
            total_count = len(accuracy_df)
            correct_pct = (correct_count / total_count * 100) if total_count > 0 else 0
            
            # Calculate accuracy by confidence level if confidence column exists
            if 'confidence' in accuracy_df.columns:
                st.metric("Overall Accuracy", f"{correct_pct:.1f}% ({correct_count}/{total_count} correct)")
                
                # Group by confidence level to show accuracy for each level
                confidence_metrics = accuracy_df.groupby('confidence')['correct'].agg(['mean', 'count', 'sum']).reset_index()
                confidence_metrics['accuracy'] = confidence_metrics['mean'] * 100
                
                # Display accuracy by confidence level
                st.subheader("Accuracy by Confidence Level")
                for _, row in confidence_metrics.iterrows():
                    conf_level = row['confidence']
                    conf_acc = row['accuracy']
                    conf_count = row['count']
                    true_count = row['sum']
                    st.metric(
                        f"{conf_level} Predictions ({conf_count})", 
                        f"{conf_acc:.1f}% ({true_count}/{conf_count} correct)"
                    )
            else:
                # Just display the overall metric if no confidence data
                st.metric("Accuracy", f"{correct_pct:.1f}%")
            
            # Reorder columns to show confidence prominently
            cols = ['period', 'predicted', 'actual', 'correct', 'confidence']
            display_cols = [col for col in cols if col in accuracy_df.columns]
            ordered_df = accuracy_df[display_cols]
            
            # Use our safe display function to avoid Arrow serialization issues
            display_df = safe_dataframe_display(ordered_df, correct_col='correct')
                
            st.dataframe(display_df)
    
    # Show raw data
    if show_raw_data:
        st.header("Raw Data")
        st.dataframe(df.head(50))
    
    # Last updated info
    st.sidebar.info(f"Last updated: {st.session_state.last_update_time}")
    
    # Set auto-refresh - always enabled
    if st.session_state.auto_refresh:
        time.sleep(10)  # Small delay to prevent excessive reruns
        st.rerun()
# Display data status message
if 'df' in locals() and df is not None and not df.empty:
    print(f"DataFrame is loaded with {len(df)} records")
    st.success(f"Data loaded successfully. Found {len(df)} game records.")
else:
    print("DataFrame is not properly loaded or is empty")
    if len(df) == 0:
        st.info("No game data available yet. Use the form below to add new game data.")
    # Ensure we have a properly structured DataFrame even if empty
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(columns=['period', 'price', 'number', 'color', 'timestamp'])

# Create directory for models if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load prediction history from Supabase
load_prediction_history()
