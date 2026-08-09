from src.components.data_preprocessing import run_preprocessing
from src.components.feature_engineering import run_feature_engineering
from src.components.model_training import run_model_training
from src.components.evaluation import run_evaluation


def run_pipeline():
    print("Pipeline started...")

    run_preprocessing()
    print("Data preprocessing completed!")

    X_train, X_test, y_train, y_test = run_feature_engineering()
    print("Feature engineering completed!")

    best_model, scores = run_model_training(X_train, X_test, y_train, y_test)
    print(f"Model training completed! Best model: {best_model}")

    run_evaluation(X_test, y_test)