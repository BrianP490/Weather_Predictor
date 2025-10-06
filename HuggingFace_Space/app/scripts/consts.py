"""
This module defines constants for file paths and feature names used in the application."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_DIRECTORY = 'model_weights'
MODEL_WEIGHTS_FILE_NAME = 'trained-model.pt'
MODEL_WEIGHTS_FULL_PATH = BASE_DIR / MODEL_DIRECTORY / MODEL_WEIGHTS_FILE_NAME

CONFIG_DIRECTORY = './configs'
CONFIG_FILE_NAME = 'config.json'
CONFIG_PATH = BASE_DIR / CONFIG_DIRECTORY / CONFIG_FILE_NAME

SCALER_DIRECTORY = './scalers'
FEATURE_SCALER_FILE_NAME = 'feature-scaler.joblib'
FEATURE_SCALER_PATH = BASE_DIR / SCALER_DIRECTORY / FEATURE_SCALER_FILE_NAME


# Same feature order names and order as during the model training data set
FEATURE_NAMES = ['DAY_OF_YEAR', 'PRECIPITATION', 'LAGGED_PRECIPITATION', 'AVG_WIND_SPEED', 'MIN_TEMP']