import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from src.config import PROCESSED_DATA_PATH, TEST_SIZE, RANDOM_STATE
from src.logger import get_logger

logger = get_logger(__name__)


def load_processed_data():
    logger.info("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    logger.info(f"Processed data shape: {df.shape}")
    return df


def split_data(df):
    logger.info("Splitting data into train and test...")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,   # VERY IMPORTANT
        random_state=RANDOM_STATE
    )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train):
    logger.info("Applying SMOTE to training data...")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info(f"After SMOTE: {X_resampled.shape}")
    return X_resampled, y_resampled


def run_feature_engineering():
    df = load_processed_data()

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_res, y_train_res = apply_smote(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test