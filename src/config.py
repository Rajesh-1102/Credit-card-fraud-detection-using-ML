from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

LOG_DIR = BASE_DIR / "logs"

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42