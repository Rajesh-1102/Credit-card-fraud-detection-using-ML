from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib

from xgboost import XGBClassifier

from src.config import MODEL_PATH
from src.logger import get_logger

logger = get_logger(__name__)


def train_models(X_train, y_train):
    logger.info("Training models...")

    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(eval_metric='logloss')
    }

    trained_models = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_models(models, X_test, y_test):
    logger.info("Evaluating models...")

    scores = {}

    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_pred_proba)
        scores[name] = score

        logger.info(f"{name} ROC-AUC: {score:.4f}")

    return scores


def save_best_model(models, scores):
    best_model_name = max(scores, key=scores.get)
    best_model = models[best_model_name]

    joblib.dump(best_model, MODEL_PATH)

    logger.info(f"Best model: {best_model_name}")
    logger.info("Model saved successfully!")

    return best_model_name


def run_model_training(X_train, X_test, y_train, y_test):
    models = train_models(X_train, y_train)
    scores = evaluate_models(models, X_test, y_test)
    best_model_name = save_best_model(models, scores)

    return best_model_name, scores