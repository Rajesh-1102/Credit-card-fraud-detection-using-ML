import os
os.makedirs("artifacts/plots", exist_ok=True)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc
)

from src.config import MODEL_PATH
from src.logger import get_logger

logger = get_logger(__name__)


def load_model():
    logger.info("Loading trained model...")
    model = joblib.load(MODEL_PATH)
    return model


def evaluate_model(X_test, y_test):
    model = load_model()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification Report
    report = classification_report(y_test, y_pred)

    logger.info("\n" + report)

    return y_pred, y_proba, cm


def plot_confusion_matrix(cm):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("artifacts/plots/confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig("artifacts/plots/roc_curve.png")
    plt.close()


def plot_precision_recall(y_test, y_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig("artifacts/plots/precision_recall.png")
    plt.close()


def run_evaluation(X_test, y_test):
    y_pred, y_proba, cm = evaluate_model(X_test, y_test)

    plot_confusion_matrix(cm)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall(y_test, y_proba)

    print("Evaluation completed! Plots saved.")