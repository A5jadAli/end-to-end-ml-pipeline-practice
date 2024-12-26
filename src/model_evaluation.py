import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import os
import json
import pickle
import logging
from typing import Dict, Tuple
import yaml
from dvclive import Live

# Ensure the "logs" directory exists
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger("model_evaluation")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

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

def load_model(model_path: str):
    """Load the trained model."""
    try:
        model_file_path = os.path.join(model_path, "random_forest_model.pkl")
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded successfully from %s', model_file_path)
        return model
    except Exception as e:
        logger.error('Error occurred while loading the model: %s', e)
        raise

def prepare_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare test data for evaluation."""
    try:
        # Separate features and target
        target_column = 'out'
        feature_columns = [col for col in test_data.columns if col != target_column]
        
        X_test = test_data[feature_columns].values
        y_test = test_data[target_column].values
        
        logger.debug('Test data prepared successfully')
        return X_test, y_test
    except Exception as e:
        logger.error('Error occurred while preparing test data: %s', e)
        raise

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
    """Evaluate the model using multiple metrics."""
    try:
        # Get predictions and probability scores
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_pred_proba))
        }
        
        logger.debug('Model evaluation completed successfully')
        logger.info('Model Metrics:')
        for metric, value in metrics.items():
            logger.info(f'{metric}: {value:.4f}')
        
        return metrics
    except Exception as e:
        logger.error('Error occurred while evaluating the model: %s', e)
        raise

def save_metrics(metrics: Dict, report_path: str) -> None:
    """Save the evaluation metrics as JSON."""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs(report_path, exist_ok=True)
        
        # Save metrics to JSON file
        metrics_file_path = os.path.join(report_path, "metrics.json")
        with open(metrics_file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        
        logger.debug('Metrics saved successfully to %s', metrics_file_path)
    except Exception as e:
        logger.error('Error occurred while saving metrics: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        # Load the engineered data
        train_data, test_data = load_engineered_data(data_path='./data')
        
        # Load the trained model
        model = load_model(model_path='./models')
        
        # Prepare the test data
        X_test, y_test = prepare_data(train_data, test_data)
        
        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)
        
        # experiment tracking with DVC
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy', accuracy_score(y_test, y_test))
            live.log_metric('precision', precision_score(y_test, y_test))
            live.log_metric('recall', recall_score(y_test, y_test))
            
            live.log_params(params)
        
        # Save the metrics
        save_metrics(metrics, report_path='./reports')
        
        logger.info('Model evaluation process completed successfully')
        
    except Exception as e:
        logger.error('Failed to complete the model evaluation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()