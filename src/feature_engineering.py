import pandas as pd
import numpy as np
import os
import logging
from typing import Tuple

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger("feature_engineering")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "feature_engineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_processed_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the processed train and test datasets."""
    try:
        processed_data_path = os.path.join(data_path, "processed")
        train_data = pd.read_csv(os.path.join(processed_data_path, "train_processed.csv"))
        test_data = pd.read_csv(os.path.join(processed_data_path, "test_processed.csv"))
        logger.debug('Processed train and test data loaded successfully')
        return train_data, test_data
    except Exception as e:
        logger.error('Error occurred while loading processed data: %s', e)
        raise

def handle_missing_values(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Handle missing values in the datasets."""
    try:
        # For Glucose, BloodPressure, BMI - replace 0s with median
        zero_columns = ['glu', 'bp', 'bmi']
        
        train_processed = train_data.copy()
        test_processed = test_data.copy()
        
        for column in zero_columns:
            # Calculate median excluding zeros from training data
            median_value = train_data[train_data[column] != 0][column].median()
            
            # Replace zeros with median in both datasets
            train_processed[column] = train_processed[column].replace(0, median_value)
            test_processed[column] = test_processed[column].replace(0, median_value)
        
        # For Insulin and SkinThickness - replace 0s with median
        for column in ['ins', 'st']:
            median_value = train_data[train_data[column] != 0][column].median()
            train_processed[column] = train_processed[column].replace(0, median_value)
            test_processed[column] = test_processed[column].replace(0, median_value)
        
        logger.debug('Missing values handled successfully')
        return train_processed, test_processed
    except Exception as e:
        logger.error('Error occurred while handling missing values: %s', e)
        raise

def save_engineered_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the feature engineered train and test datasets."""
    try:
        engineered_data_path = os.path.join(data_path, "engineered")
        os.makedirs(engineered_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(engineered_data_path, "train_engineered.csv"), index=False)
        test_data.to_csv(os.path.join(engineered_data_path, "test_engineered.csv"), index=False)
        
        logger.debug('Engineered train and test data saved to %s', engineered_data_path)
    except Exception as e:
        logger.error('Error occurred while saving engineered data: %s', e)
        raise

def main():
    try:
        # Load the processed train and test data
        train_data, test_data = load_processed_data(data_path='./data')
        
        # Handle missing values
        train_data, test_data = handle_missing_values(train_data, test_data)
        
        # Save the engineered data
        save_engineered_data(train_data, test_data, data_path='./data')
        
        logger.debug('Feature engineering completed successfully')
        
    except Exception as e:
        logger.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()