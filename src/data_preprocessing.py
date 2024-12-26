import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import logging
from typing import Tuple

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_train_test_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        train_data = pd.read_csv(os.path.join(raw_data_path, "train.csv"))
        test_data = pd.read_csv(os.path.join(raw_data_path, "test.csv"))
        logger.debug('Train and test data loaded successfully')
        return train_data, test_data
    except Exception as e:
        logger.error('Error occurred while loading train/test data: %s', e)
        raise

def preprocess_features(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scale the features using StandardScaler."""
    try:
        # Initialize the scaler
        scaler = StandardScaler()
        
        # Get feature columns (all except 'out')
        feature_columns = [col for col in train_data.columns if col != 'out']
        
        # Fit scaler on train data and transform both train and test
        train_scaled = train_data.copy()
        test_scaled = test_data.copy()
        
        train_scaled[feature_columns] = scaler.fit_transform(train_data[feature_columns])
        test_scaled[feature_columns] = scaler.transform(test_data[feature_columns])
        
        logger.debug('Feature scaling completed successfully')
        return train_scaled, test_scaled
    except Exception as e:
        logger.error('Error occurred during feature scaling: %s', e)
        raise

def encode_target(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Encode the target variable (although not needed for binary classification in this case)."""
    try:
        # Initialize the encoder
        encoder = LabelEncoder()
        
        # Fit encoder on train data and transform both train and test
        train_encoded = train_data.copy()
        test_encoded = test_data.copy()
        
        train_encoded['out'] = encoder.fit_transform(train_data['out'])
        test_encoded['out'] = encoder.transform(test_data['out'])
        
        logger.debug('Target encoding completed successfully')
        return train_encoded, test_encoded
    except Exception as e:
        logger.error('Error occurred during target encoding: %s', e)
        raise

def save_processed_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the preprocessed train and test datasets."""
    try:
        processed_data_path = os.path.join(data_path, "processed")
        os.makedirs(processed_data_path, exist_ok=True)
        
        train_data.to_csv(os.path.join(processed_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(processed_data_path, "test_processed.csv"), index=False)
        
        logger.debug('Processed train and test data saved to %s', processed_data_path)
    except Exception as e:
        logger.error('Error occurred while saving processed data: %s', e)
        raise

def main():
    try:
        # Load the train and test data
        train_data, test_data = load_train_test_data(data_path='./data')
        
        # Scale the features
        train_scaled, test_scaled = preprocess_features(train_data, test_data)
        
        # Encode the target (though not strictly necessary for binary classification)
        train_processed, test_processed = encode_target(train_scaled, test_scaled)
        
        # Save the processed data
        save_processed_data(train_processed, test_processed, data_path='./data')
        
        logger.info('Data preprocessing completed successfully')
        
    except Exception as e:
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()