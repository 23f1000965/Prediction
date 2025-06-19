import os
import pandas as pd
import datetime

def save_data_to_csv(data_dict, file_path='data/wingo_parity_data.csv'):
    """
    Save new data to CSV file with newest entries first.
    
    Args:
        data_dict: Dictionary containing the new data entry
        file_path: Path to the CSV file
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

def add_new_game_data(period, price, number, color, timestamp=None):
    """
    Add a new game result to the CSV file.
    
    Args:
        period: Game period ID
        price: Game price
        number: Result number
        color: Result color
        timestamp: Optional timestamp (defaults to current time)
    
    Returns:
        Boolean indicating success
    """
    if timestamp is None:
        timestamp = datetime.datetime.now()
        
    # Create data dictionary
    data = {
        'period': str(period),
        'price': float(price),
        'number': str(number),
        'color': str(color),
        'timestamp': timestamp
    }
    
    # Save to CSV
    return save_data_to_csv(data)
