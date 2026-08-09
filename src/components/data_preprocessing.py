import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, SCALER_PATH
from src.logger import get_logger

logger = get_logger(__name__)


def load_data():
    logger.info("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    logger.info(f"Data loaded with shape: {df.shape}")
    return df


def clean_data(df):
    logger.info("Cleaning data...")

    # Check null values
    if df.isnull().sum().sum() > 0:
        logger.warning("Null values found. Dropping...")
        df = df.dropna()

    # Drop duplicates
    df = df.drop_duplicates()

    logger.info(f"Data after cleaning: {df.shape}")
    return df


def scale_features(df):
    logger.info("Scaling features...")

    scaler = StandardScaler()

    # Exclude target column
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_scaled = scaler.fit_transform(X)

    # Convert back to DataFrame
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Combine back
    df_scaled = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    logger.info("Scaling completed")
    return df_scaled, scaler

from pathlib import Path

def save_outputs(df, scaler):
    logger.info("Saving processed data and scaler...")

    # Create directories if they don't exist
    Path(PROCESSED_DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(SCALER_PATH).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)

    logger.info("Saved successfully!")


def run_preprocessing():
    df = load_data()
    df = clean_data(df)
    df_scaled, scaler = scale_features(df)
    save_outputs(df_scaled, scaler)

    return df_scaled