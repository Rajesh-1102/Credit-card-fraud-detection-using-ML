from src.database import insert_transaction
import pandas as pd
import joblib

from src.config import MODEL_PATH, SCALER_PATH
from src.logger import get_logger

logger = get_logger(__name__)


class FraudPredictor:
    def __init__(self):
        logger.info("Loading model and scaler...")
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

    def preprocess(self, input_data):
        """
        input_data: dict or DataFrame
        """
        df = pd.DataFrame([input_data])
        
        feature_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df = df[feature_order]
        # Scale features
        df_scaled = self.scaler.transform(df)

        return df_scaled

    def predict(self, input_data):
        df_scaled = self.preprocess(input_data)

        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0][1]
        
        insert_transaction(int(prediction), float(probability))
        result = {
            "prediction": int(prediction),
            "probability": float(probability)
        }

        return result