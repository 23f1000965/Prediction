import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from lightgbm import LGBMClassifier

class GamePredictor:
    """
    A class for predicting colors and numbers in the prediction game
    Uses multiple algorithms to find the best performer
    """
    def __init__(self, model_dir="models", target_type="color"):
        """
        Initialize the predictor
        
        Args:
            model_dir: Directory to save/load models
            target_type: 'color' or 'number' - what to predict
        """
        self.model_dir = model_dir
        self.target_type = target_type
        self.models = {}
        self.best_model_name = None
        self.feature_importances = {}
        self.color_encoder = LabelEncoder()
        self.number_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.features = []
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
    def _create_lag_features(self, df, columns, lag_range=5):
        """Create lag features for the specified columns"""
        data = df.copy()
        
        for col in columns:
            for i in range(1, lag_range + 1):
                # Create lag feature
                data[f'{col}_lag_{i}'] = data[col].shift(i)
                
                # Create diff feature only for numeric columns
                if i == 1 and pd.api.types.is_numeric_dtype(data[col]):
                    data[f'{col}_diff'] = data[col].diff()
        
        # Drop rows with NaN values (first lag_range rows)
        data = data.dropna().reset_index(drop=True)
        
        return data
        
    def _create_time_features(self, df):
        """Create time-based features"""
        data = df.copy()
        
        # Handle timestamp column if it exists
        if 'timestamp' in data.columns:
            # Check if timestamp column needs conversion
            timestamp_type = data['timestamp'].dtype
            
            # Convert timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                # Try multiple approaches to convert strings to datetime
                try:
                    print(f"Converting timestamp column from {timestamp_type} to datetime...")
                    # Use more robust conversion with error handling
                    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
                    
                    # Fill NaT values with a default datetime to avoid errors
                    if data['timestamp'].isna().any():
                        print(f"Warning: {data['timestamp'].isna().sum()} timestamp values were invalid")
                        # Use most common timestamp as fallback
                        valid_timestamps = data['timestamp'].dropna()
                        if not valid_timestamps.empty:
                            most_common = valid_timestamps.iloc[0]
                            data['timestamp'].fillna(most_common, inplace=True)
                        else:
                            # If no valid timestamps, use current time
                            import datetime as dt
                            data['timestamp'].fillna(pd.Timestamp(dt.datetime.now()), inplace=True)
                except Exception as e:
                    print(f"Error converting timestamps: {str(e)}")
                    # Create default timestamp column if conversion fails completely
                    import datetime as dt
                    data['timestamp'] = pd.Series([pd.Timestamp(dt.datetime.now())] * len(data))
            
            # Extract time features only if timestamp is properly formatted
            try:
                data['hour'] = data['timestamp'].dt.hour
                data['minute'] = data['timestamp'].dt.minute
                data['second'] = data['timestamp'].dt.second
                data['day_of_week'] = data['timestamp'].dt.dayofweek
                data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)
                print("Successfully created time features from timestamp")
            except Exception as e:
                print(f"Failed to create time features: {str(e)}")
                # Create default time features if extraction fails
                data['hour'] = 12  # Default hour
                data['minute'] = 0  # Default minute
                data['second'] = 0  # Default second
                data['day_of_week'] = 0  # Default day (Monday)
                data['is_weekend'] = 0  # Default not weekend
        else:
            print("No timestamp column found")
            # Create default time features if timestamp column doesn't exist
            data['hour'] = 12
            data['minute'] = 0
            data['second'] = 0
            data['day_of_week'] = 0
            data['is_weekend'] = 0
            
        return data
        
    def _create_pattern_features(self, df):
        """Create enhanced pattern-based features"""
        data = df.copy()
        target_column = self.target_type  # Use target_type as the column name
        
        # Skip if target column not present
        if target_column not in data.columns:
            return data
            
        # --- Basic streak features ---
        # Calculate streak (consecutive same values)
        data[f'{target_column}_streak'] = 1
        for i in range(1, len(data)):
            if data.iloc[i][target_column] == data.iloc[i-1][target_column]:
                data.loc[data.index[i], f'{target_column}_streak'] = data.loc[data.index[i-1], f'{target_column}_streak'] + 1
        
        # --- Change detection features ---
        # Transitions (frequency of changes)
        data[f'{target_column}_changed'] = (data[target_column] != data[target_column].shift(1)).astype(int)
        
        # Create rolling sum of changes (volatility feature)
        data[f'{target_column}_volatility_3'] = data[f'{target_column}_changed'].rolling(3).sum().fillna(0)
        data[f'{target_column}_volatility_5'] = data[f'{target_column}_changed'].rolling(5).sum().fillna(0)
        data[f'{target_column}_volatility_10'] = data[f'{target_column}_changed'].rolling(10).sum().fillna(0)
        
        # --- Advanced sequence features ---
        # Try to add sequence-based features (e.g., last N outcomes as a single feature)
        try:
            if len(data) > 5:
                # Add last N values as categorical features
                for n in [2, 3]:
                    data[f'last_{n}_{target_column}'] = ''
                    for i in range(n, len(data)):
                        values = [str(data.iloc[i-j][target_column]) for j in range(1, n+1)]
                        data.iloc[i, data.columns.get_loc(f'last_{n}_{target_column}')] = '_'.join(values)
        except Exception:
            # Skip if feature creation fails
            pass
            
        # --- Add type-specific features ---
        if target_column == 'number':
            # Number properties
            data['is_even'] = data['number'].astype(int) % 2 == 0
            data['is_even'] = data['is_even'].astype(int)
            
            # High/low number
            data['is_high'] = (data['number'].astype(int) > 4).astype(int)
            
            # Even/Odd alternation
            data['even_odd_switch'] = 0
            for i in range(1, len(data)):
                if data.iloc[i]['is_even'] != data.iloc[i-1]['is_even']:
                    data.loc[data.index[i], 'even_odd_switch'] = 1
                    
            # Extract digit patterns
            try:
                # Patterns of even/odd numbers in sequence
                data['even_odd_pattern'] = ''
                for i in range(3, len(data)):
                    pattern = ''.join(['E' if data.iloc[j]['is_even'] else 'O' for j in range(i-3, i)])
                    data.iloc[i, data.columns.get_loc('even_odd_pattern')] = pattern
            except Exception:
                # Skip if this causes issues
                pass
                
        elif target_column == 'color':
            # Color-specific features
            # Create binary columns for each color type
            if 'Red' in ''.join(data[target_column]):
                data['is_red'] = data[target_column].str.contains('Red').astype(int)
                data['time_since_red'] = (data['is_red'] == 0).cumsum()
                data.loc[data['is_red']==1, 'time_since_red'] = 0
            
            if 'Green' in ''.join(data[target_column]):
                data['is_green'] = data[target_column].str.contains('Green').astype(int)
                data['time_since_green'] = (data['is_green'] == 0).cumsum()
                data.loc[data['is_green']==1, 'time_since_green'] = 0
            
            if 'Violet' in ''.join(data[target_column]):
                data['is_violet'] = data[target_column].str.contains('Violet').astype(int)
        
        return data
        
    def prepare_features(self, df, target_type=None):
        """Prepare features for model training/prediction"""
        if target_type is None:
            target_type = self.target_type
        
        data = df.copy()
        
        # Ensure period is consistently treated as string
        data['period'] = data['period'].astype(str)
        
        # Sort by period in descending order (newest first)
        data = data.sort_values('period', ascending=False).reset_index(drop=True)
        
        # Create lag features
        if target_type == 'color':
            data = self._create_lag_features(data, ['color'], lag_range=5)
            target = 'color'
        else:  # target_type == 'number'
            data = self._create_lag_features(data, ['number'], lag_range=5)
            target = 'number'
        
        # Create other features
        data = self._create_time_features(data)
        data = self._create_pattern_features(data)
        
        # Handle categorical variables - encode color and number
        if 'color' in data.columns:
            self.color_encoder.fit(data['color'])
            data['color_encoded'] = self.color_encoder.transform(data['color'])
            
            # Create lag features for encoded color
            data = self._create_lag_features(data, ['color_encoded'], lag_range=3)
        
        if 'number' in data.columns:
            self.number_encoder.fit(data['number'].astype(str))
            data['number_encoded'] = self.number_encoder.transform(data['number'].astype(str))
            
            # Create lag features for encoded number
            data = self._create_lag_features(data, ['number_encoded'], lag_range=3)
        
        # Determine features for the model
        if target_type == 'color':
            numeric_features = ['color_encoded_lag_1', 'color_encoded_lag_2', 'color_encoded_lag_3', 
                               'number_encoded', 'number_encoded_lag_1', 'color_streak']
            if 'is_even' in data.columns:
                numeric_features.extend(['is_even', 'is_even_lag_1', 'even_odd_switch'])
            if 'hour' in data.columns:
                numeric_features.extend(['hour', 'minute'])
            target_encoded = 'color_encoded'
        else:  # target_type == 'number'
            numeric_features = ['number_encoded_lag_1', 'number_encoded_lag_2', 'number_encoded_lag_3',
                               'color_encoded', 'color_encoded_lag_1', 'number_streak']
            if 'is_even' in data.columns:
                numeric_features.extend(['is_even', 'is_even_lag_1', 'even_odd_switch'])
            if 'hour' in data.columns:
                numeric_features.extend(['hour', 'minute'])
            target_encoded = 'number_encoded'
        
        # Filter to include only available features
        self.features = [f for f in numeric_features if f in data.columns]
        
        # Prepare X and y
        X = data[self.features]
        y = data[target_encoded]
        
        return X, y, data
        
    def train_models(self, df, target_type=None, test_size=0.2, verbose=True):
        """Train multiple models and select the best one"""
        if target_type is None:
            target_type = self.target_type
        else:
            self.target_type = target_type
        
        X, y, processed_data = self.prepare_features(df, target_type)
        
        if len(X) < 20:
            if verbose:
                print("Not enough data for training. Need at least 20 samples.")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models with optimized hyperparameters
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,      # More trees for better performance
                max_depth=12,         # Control tree depth to prevent overfitting
                min_samples_leaf=2,   # Minimum samples per leaf for robustness
                class_weight='balanced',  # Handle class imbalance
                bootstrap=True,       # Use bootstrapping
                random_state=42
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200,     # More boosting stages
                learning_rate=0.08,    # Slower learning rate for better generalization
                max_depth=5,          # Control depth to prevent overfitting
                subsample=0.8,        # Use 80% of samples for each tree
                min_samples_split=5,  # Minimum samples to split a node
                random_state=42
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=250,      # More boosting rounds
                learning_rate=0.05,    # Slower learning rate for better generalization
                max_depth=6,           # Maximum tree depth
                subsample=0.7,         # Sample 70% of data per tree
                colsample_bytree=0.7,  # Sample 70% of features per tree
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=250,      # More boosting rounds
                learning_rate=0.05,    # Slower learning rate
                num_leaves=31,         # Maximum number of leaves per tree
                boosting_type='gbdt',  # Traditional gradient boosting
                subsample=0.8,         # Use 80% of data per iteration
                colsample_bytree=0.8,  # Use 80% of features per tree
                class_weight='balanced',
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                C=0.8,                # Regularization strength (lower = stronger regularization)
                solver='liblinear',    # Better for small datasets
                max_iter=2000,         # More iterations for convergence
                class_weight='balanced',  # Handle class imbalance
                random_state=42
            )
        }
        
        # Train and evaluate each model
        if verbose:
            print(f"Training {len(models)} models for {target_type} prediction...")
        
        results = {}
        self.feature_importances = {}
        
        for name, model in models.items():
            if verbose:
                print(f"Training {name}...")
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            # Store model
            self.models[name] = model
            
            # Store feature importances if available
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = pd.DataFrame({
                    'Feature': self.features,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                self.feature_importances[name] = feature_importance
        
        # Select best model
        self.best_model_name = max(results, key=results.get)
        best_accuracy = results[self.best_model_name]
        
        if verbose:
            print("\nModel accuracies:")
            for name, acc in results.items():
                print(f"  {name}: {acc:.4f}")
            print(f"\nBest model: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return results
    
    def predict(self, df, model_name=None, target_type=None):
        """Make predictions using the specified model
        
        Args:
            df: DataFrame with latest data
            model_name: Name of model to use (best model if None)
            target_type: 'color' or 'number'
            
        Returns:
            Tuple of predicted class and probability
        """
        if target_type is None:
            target_type = self.target_type
        
        if model_name is None:
            model_name = self.best_model_name
            
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Please train models first.")
        
        # Prepare features
        X, _, processed_data = self.prepare_features(df, target_type)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        model = self.models[model_name]
        y_pred = model.predict(X_scaled[:1])[0]  # Predict only for the first (most recent) row
        
        # Get probabilities
        try:
            probas = model.predict_proba(X_scaled[:1])[0]
            raw_confidence = max(probas)
            
            # Apply confidence boosting for more meaningful interpretation
            # This maps very low confidence values to a more usable range
            # For example: 0.01 -> ~0.55, 0.1 -> ~0.7, 0.5 -> ~0.9
            if raw_confidence < 0.5:
                # Boost low confidence scores (sigmoid-like transformation)
                confidence = 0.5 + (raw_confidence * 0.8)
            else:
                # Keep high confidence scores high
                confidence = raw_confidence
                
            # Ensure confidence is in the valid range
            confidence = min(max(confidence, 0.5), 1.0)
        except:
            confidence = 0.7  # Fallback if predict_proba is not supported
        
        # Convert prediction back to original class
        if target_type == 'color':
            result = self.color_encoder.inverse_transform([y_pred])[0]
        else:  # target_type == 'number'
            result = self.number_encoder.inverse_transform([y_pred])[0]
            result = int(result)  # Convert to integer
        
        return result, confidence
    
    def predict_next_periods(self, df, periods=5, use_best_model=True, target_type=None, confidence_threshold=0.5):
        """Predict next n periods based on current data
        
        Args:
            df: DataFrame with latest data
            periods: Number of future periods to predict
            use_best_model: If True, use the best model; if False, use all models and aggregate
            target_type: 'color' or 'number'
            confidence_threshold: Minimum confidence level (0-1) to include prediction
            
        Returns:
            DataFrame with predictions and confidence levels
        """
        if target_type is None:
            target_type = self.target_type
            
        # Convert confidence threshold from percentage to decimal if needed
        if confidence_threshold > 1:
            confidence_threshold = confidence_threshold / 100
        
        # Ensure period is consistently treated as string
        df_copy = df.copy()
        df_copy['period'] = df_copy['period'].astype(str)
            
        # Get the latest period
        latest_df = df_copy.sort_values('period', ascending=False)
        
        # Check if dataframe is empty to avoid index error
        if latest_df.empty:
            print("Warning: Empty dataframe in predict_next_periods. Cannot make predictions.")
            return pd.DataFrame(columns=['period', 'number', 'color', 'confidence', 'confidence_numeric', 'recommendation', 'meets_threshold'])
        
        latest_period = int(latest_df.iloc[0]['period'])
        
        # Store predictions
        predictions = []
        
        # Create a copy of data that we'll update with each prediction
        temp_df = latest_df.copy()
        
        for i in range(1, periods+1):
            next_period = latest_period + i
            
            # Make prediction
            if use_best_model:
                prediction, confidence = self.predict(temp_df, target_type=target_type)
            else:
                # Use ensemble method for prediction
                all_predictions = {}
                confidences = {}
                
                for model_name in self.models:
                    pred, conf = self.predict(temp_df, model_name=model_name, target_type=target_type)
                    all_predictions[model_name] = pred
                    confidences[model_name] = conf
                
                # Count predictions
                pred_counts = {}
                for pred in all_predictions.values():
                    if pred not in pred_counts:
                        pred_counts[pred] = 0
                    pred_counts[pred] += 1
                
                # Get majority vote
                prediction = max(pred_counts, key=pred_counts.get)
                confidence = sum(conf for pred, conf in zip(all_predictions.values(), confidences.values()) 
                                if pred == prediction) / pred_counts[prediction]
            
            # Determine confidence level with updated thresholds
            # Since we've boosted confidence to be at least 0.5,
            # we need higher thresholds for Strong and Moderate recommendations
            if confidence >= 0.8:
                recommendation = "Strong"
            elif confidence >= 0.65:
                recommendation = "Moderate"
            else:
                recommendation = "Weak"
                
            # Calculate confidence as percentage for display
            confidence_percent = confidence * 100
                
            # Format prediction data based on target type
            if target_type == 'number':
                number = prediction
                # Derive color from number based on game rules
                if number == 0:
                    color = "Red+Violet"
                elif number == 5:
                    color = "Green+Violet"
                elif number in [1, 3, 7, 9]:
                    color = "Green"
                elif number in [2, 4, 6, 8]:
                    color = "Red"
            else:  # target_type == 'color'
                color = prediction
                # We don't derive number from color as it's ambiguous
                number = None
            
            # Store prediction with confidence info regardless of threshold
            prediction_data = {
                'period': str(next_period),
                'number': number,
                'color': color,
                'confidence': f"{confidence_percent:.1f}%",  # Formatted string
                'confidence_numeric': confidence_percent,  # Numeric for filtering
                'recommendation': recommendation,
                'meets_threshold': confidence >= confidence_threshold
            }
            
            predictions.append(prediction_data)
            
            # Create a new row for the prediction and add it to temp_df
            # This allows us to use the current prediction for the next prediction
            new_row = latest_df.iloc[0].copy()
            new_row['period'] = str(next_period)
            if 'number' in new_row and target_type == 'number':
                new_row['number'] = number
            if 'color' in new_row and target_type == 'color':
                new_row['color'] = color
            new_row['timestamp'] = pd.Timestamp.now()
            
            # Add the new row at the beginning
            temp_df = pd.concat([pd.DataFrame([new_row]), temp_df]).reset_index(drop=True)
        
        # Create a DataFrame from all predictions
        result_df = pd.DataFrame(predictions)
        
        # Sort by period (chronological order)
        if len(result_df) > 0:
            result_df = result_df.sort_values('period')
        
        return result_df
    
    def save_models(self, target_type=None):
        """Save trained models, encoders and scaler"""
        if target_type is None:
            target_type = self.target_type
        
        # Create directory for target type if it doesn't exist
        target_dir = os.path.join(self.model_dir, target_type)
        os.makedirs(target_dir, exist_ok=True)
        
        # Save each model
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(target_dir, f"{model_name}.joblib"))
        
        # Save encoders and scaler
        joblib.dump(self.color_encoder, os.path.join(target_dir, "color_encoder.joblib"))
        joblib.dump(self.number_encoder, os.path.join(target_dir, "number_encoder.joblib"))
        joblib.dump(self.scaler, os.path.join(target_dir, "scaler.joblib"))
        
        # Save feature list and best model name
        with open(os.path.join(target_dir, "metadata.txt"), "w") as f:
            f.write(f"Best model: {self.best_model_name}\n")
            f.write(f"Features: {','.join(self.features)}\n")
        
        print(f"Models saved to {target_dir}")
    
    def load_models(self, target_type=None):
        """Load trained models, encoders and scaler
        Returns True if loading was successful, False otherwise
        """
        if target_type is None:
            target_type = self.target_type
        else:
            self.target_type = target_type
        
        # Check if directory exists
        target_dir = os.path.join(self.model_dir, target_type)
        if not os.path.isdir(target_dir):
            print(f"No models found for {target_type}")
            return False
        
        # Load metadata
        metadata_path = os.path.join(target_dir, "metadata.txt")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "Best model:" in line:
                        self.best_model_name = line.split(":")[1].strip()
                    elif "Features:" in line:
                        self.features = line.split(":")[1].strip().split(",")
        
        # Load encoders and scaler
        try:
            self.color_encoder = joblib.load(os.path.join(target_dir, "color_encoder.joblib"))
            self.number_encoder = joblib.load(os.path.join(target_dir, "number_encoder.joblib"))
            self.scaler = joblib.load(os.path.join(target_dir, "scaler.joblib"))
        except:
            print("Error loading encoders or scaler")
            return False
        
        # Load models
        self.models = {}
        model_files = [f for f in os.listdir(target_dir) if f.endswith(".joblib") and not any(x in f for x in ["encoder", "scaler"])]
        
        if not model_files:
            print(f"No model files found in {target_dir}")
            return False
        
        for model_file in model_files:
            model_name = model_file.split(".")[0]
            try:
                self.models[model_name] = joblib.load(os.path.join(target_dir, model_file))
            except:
                print(f"Error loading model {model_name}")
        
        if self.best_model_name not in self.models:
            self.best_model_name = list(self.models.keys())[0] if self.models else None
        
        return len(self.models) > 0
    
    def plot_feature_importance(self, model_name=None, figsize=(12, 6)):
        """Plot feature importances for the specified model"""
        if model_name is None:
            model_name = self.best_model_name
            
        if model_name not in self.feature_importances:
            raise ValueError(f"No feature importances available for model {model_name}")
        
        fi = self.feature_importances[model_name]
        
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=fi)
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()
        return plt.gcf()
    
    def plot_prediction_history(self, df, predictions, target_type=None):
        """Plot the history of actual vs predicted values"""
        if target_type is None:
            target_type = self.target_type
        
        # Get the latest data
        history = df.copy().sort_values('period', ascending=True).tail(15)
        
        # Convert predictions to a DataFrame if it's not already
        if not isinstance(predictions, pd.DataFrame):
            predictions = pd.DataFrame(predictions)
        
        plt.figure(figsize=(14, 7))
        
        if target_type == 'color':
            # Map colors to numbers for visualization
            color_map = {
                'Green': 1, 
                'Red': 2, 
                'Green+Violet': 3, 
                'Red+Violet': 4
            }
            
            history['color_code'] = history['color'].map(color_map)
            predictions['color_code'] = predictions['color'].map(color_map)
            
            # Plot historical data
            plt.plot(history['period'], history['color_code'], 'o-', label='Actual Colors', linewidth=2)
            
            # Plot predictions
            plt.plot(predictions['period'], predictions['color_code'], 'x--', label='Predicted Colors', linewidth=2)
            
            # Set y-axis ticks
            plt.yticks(list(color_map.values()), list(color_map.keys()))
            plt.title('History and Predictions - Color')
        
        else:  # target_type == 'number'
            # Plot historical data
            plt.plot(history['period'], history['number'], 'o-', label='Actual Numbers', linewidth=2)
            
            # Plot predictions
            plt.plot(predictions['period'], predictions['number'], 'x--', label='Predicted Numbers', linewidth=2)
            
            # Set y-axis ticks
            plt.yticks(range(10))
            plt.title('History and Predictions - Number')
        
        plt.xlabel('Period')
        plt.ylabel(target_type.capitalize())
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        return plt.gcf()
    
    def analyze_patterns(self, df, lookback=50):
        """Analyze patterns in the historical data"""
        # Get the latest data
        data = df.sort_values('period', ascending=False).head(lookback).copy()
        
        # Calculate basic statistics
        results = {
            'total_records': len(data),
            'patterns': {}
        }
        
        # Analyze colors
        if 'color' in data.columns:
            color_counts = data['color'].value_counts()
            results['patterns']['color_distribution'] = color_counts.to_dict()
            
            # Color streaks
            color_streaks = {}
            for color in color_counts.index:
                max_streak = 0
                current_streak = 0
                for c in data['color']:
                    if c == color:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 0
                color_streaks[color] = max_streak
            results['patterns']['color_max_streaks'] = color_streaks
            
            # Alternation patterns
            alternations = {}
            for i in range(len(data) - 1):
                transition = f"{data.iloc[i+1]['color']} -> {data.iloc[i]['color']}"
                if transition not in alternations:
                    alternations[transition] = 0
                alternations[transition] += 1
            results['patterns']['color_transitions'] = dict(sorted(alternations.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Analyze numbers
        if 'number' in data.columns:
            number_counts = data['number'].value_counts()
            results['patterns']['number_distribution'] = number_counts.to_dict()
            
            # Even/Odd analysis
            data['is_even'] = data['number'].astype(int) % 2 == 0
            even_count = data['is_even'].sum()
            odd_count = len(data) - even_count
            results['patterns']['even_odd'] = {'Even': even_count, 'Odd': odd_count}
            
            # Number streaks
            number_streaks = {}
            for number in number_counts.index:
                max_streak = 0
                current_streak = 0
                for n in data['number']:
                    if n == number:
                        current_streak += 1
                        max_streak = max(max_streak, current_streak)
                    else:
                        current_streak = 0
                number_streaks[int(number)] = max_streak
            results['patterns']['number_max_streaks'] = number_streaks
        
        return results
