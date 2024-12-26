import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
import logging
from typing import Tuple

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger("model_building")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "model_building.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_engineered_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the engineered train and test datasets."""
    try:
        engineered_data_path = os.path.join(data_path, "engineered")
        train_data = pd.read_csv(os.path.join(engineered_data_path, "train_engineered.csv"))
        test_data = pd.read_csv(os.path.join(engineered_data_path, "test_engineered.csv"))
        logger.debug('Engineered train and test data loaded successfully')
        return train_data, test_data
    except Exception as e:
        logger.error('Error occurred while loading engineered data: %s', e)
        raise

def prepare_train_test_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare features and target variables for training."""
    try:
        # Separate features and target
        target_column = 'out'
        feature_columns = [col for col in train_data.columns if col != target_column]
        
        X_train = train_data[feature_columns].values
        y_train = train_data[target_column].values
        
        X_test = test_data[feature_columns].values
        y_test = test_data[target_column].values
        
        logger.debug('Train and test data prepared successfully')
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error('Error occurred while preparing train/test data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """Train the Random Forest model."""
    try:
        logger.debug('Initializing RandomForest model with parameters: %s', params)
        # Initialize and train the model
        rf_model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=params['random_state']
        )
        
        rf_model.fit(X_train, y_train)
        logger.debug('Random Forest model trained successfully')
        return rf_model
    except Exception as e:
        logger.error('Error occurred while training the model: %s', e)
        raise

def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """Save the trained model."""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Save the model
        model_file_path = os.path.join(model_path, "random_forest_model.pkl")
        with open(model_file_path, 'wb') as file:
            pickle.dump(model, file)
        
        logger.debug('Model saved successfully to %s', model_file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        params = {'n_estimators':100, 'max_depth': 10, 'min_samples_split':2, 'min_samples_leaf':1, 'random_state':42}
        # Load the engineered train and test data
        train_data, test_data = load_engineered_data(data_path='./data')
        
        # Prepare the data for training
        X_train, X_test, y_train, y_test = prepare_train_test_data(train_data, test_data)
        
        # Train the model
        model = train_model(X_train, y_train, params)
        
        # Save the model
        save_model(model, model_path='./models')
        
        logger.debug('Model training process completed successfully')
        
    except Exception as e:
        logger.error('Failed to complete the model training process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()