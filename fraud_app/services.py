from functools import lru_cache
import sqlite3

from django.conf import settings
from django.utils import timezone

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

MODEL_PATH = settings.BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = settings.BASE_DIR / "models" / "scaler.pkl"
RAW_DATA_PATH = settings.BASE_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_DATA_PATH = settings.BASE_DIR / "data" / "processed" / "processed.csv"
DB_PATH = settings.BASE_DIR / "database" / "fraud.db"
CLASS_OPTIONS = {
    "all": "All transactions",
    "fraud": "Fraud only",
    "legitimate": "Legitimate only",
}

ASSISTANT_PROMPTS = [
    {"key": "overview", "label": "Project Overview", "question": "Explain project overview"},
    {"key": "dashboard", "label": "Dashboard Guide", "question": "Explain dashboard guide"},
    {"key": "predict", "label": "Prediction Process", "question": "Explain prediction process"},
    {"key": "dataset", "label": "Dataset Features", "question": "Explain dataset features"},
    {"key": "model", "label": "Machine Learning Model", "question": "Explain machine learning model"},
    {"key": "risk", "label": "Risk Assessment", "question": "Explain risk assessment"},
    {"key": "response", "label": "Fraud Response", "question": "Explain fraud response workflow"},
    {"key": "history", "label": "Prediction History", "question": "Explain prediction history"},
    {"key": "performance", "label": "Performance Metrics", "question": "Explain model performance metrics"},
    {"key": "reports", "label": "Reports", "question": "Explain transaction reports"},
    {"key": "system", "label": "System Architecture", "question": "Explain system architecture"},
    {"key": "faq", "label": "Frequently Asked Questions", "question": "Explain frequently asked questions"},
]


class ProjectDependencyError(RuntimeError):
    pass


def _import_or_raise(package_name, import_name=None):
    try:
        module = __import__(import_name or package_name)
    except ImportError as exc:
        raise ProjectDependencyError(
            f"Could not import {package_name}: {exc}. "
            "Install dependencies with `python -m pip install -r requirements.txt`."
        ) from exc
    return module


@lru_cache(maxsize=1)
def load_model():
    joblib = _import_or_raise("joblib")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_scaler():
    joblib = _import_or_raise("joblib")
    return joblib.load(SCALER_PATH)


def normalize_features(payload):
    features = {}
    for name in FEATURE_ORDER:
        raw_value = payload.get(name, 0)
        if raw_value in ("", None):
            raw_value = 0
        try:
            features[name] = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number.") from exc
    return features


def predict_transaction(payload, save=True):
    pandas = _import_or_raise("pandas")
    features = normalize_features(payload)
    df = pandas.DataFrame([features], columns=FEATURE_ORDER)

    scaler = load_scaler()
    model = load_model()
    scaled = scaler.transform(df)

    prediction = int(model.predict(scaled)[0])
    probability = float(model.predict_proba(scaled)[0][1])

    # Generate custom values for banking simulation
    import random
    import string
    now = timezone.localtime()
    date_str = now.strftime("%Y%m%d")
    random_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    prefix = "FRD" if prediction else "TXN"
    reference_id = f"{prefix}-{date_str}-{random_suffix}"

    risk_lvl = risk_category(probability)
    status = "Blocked" if prediction else "Approved"
    action_taken = "Transaction temporarily blocked" if prediction else "Transaction approved"
    created_date = now.strftime("%Y-%m-%d")
    created_time = now.strftime("%H:%M:%S")

    tx_id = None
    if save:
        tx_id = save_transaction(
            prediction,
            probability,
            reference_id=reference_id,
            risk_level=risk_lvl,
            action_taken=action_taken,
            status=status,
            created_date=created_date,
            created_time=created_time,
            amount=features.get("Amount", 120.00),
        )

    return {
        "id": tx_id,
        "prediction": prediction,
        "label": "Fraud" if prediction else "Legitimate",
        "probability": probability,
        "risk_category": risk_lvl,
        "risk_message": risk_message(prediction, probability),
        "features": features,
        "reference_id": reference_id,
        "status": status,
        "action_taken": action_taken,
        "created_date": created_date,
        "created_time": created_time,
    }



def risk_category(probability):
    if probability >= 0.75:
        return "High risk"
    if probability >= 0.35:
        return "Medium risk"
    return "Low risk"


def risk_message(prediction, probability):
    if prediction:
        return "Review this transaction before approval and verify customer identity."
    if probability >= 0.35:
        return "Prediction is legitimate, but the score is elevated enough for analyst review."
    return "Transaction pattern looks consistent with legitimate activity."


def _dataset_path():
    path = RAW_DATA_PATH if RAW_DATA_PATH.exists() else PROCESSED_DATA_PATH
    if not path.exists():
        raise FileNotFoundError("Credit card dataset was not found in data/raw or data/processed.")
    return path


def load_transaction_dataset(usecols=None):
    pandas = _import_or_raise("pandas")
    return pandas.read_csv(_dataset_path(), usecols=usecols)


def get_sample_choices():
    df = load_transaction_dataset(usecols=FEATURE_ORDER + ["Class"])
    choices = []

    sample_specs = [
        ("legitimate", "Legitimate sample", df[df["Class"] == 0].head(3)),
        ("fraud", "Fraud sample", df[df["Class"] == 1].head(3)),
        ("high", "High amount sample", df.sort_values("Amount", ascending=False).head(3)),
    ]

    for group, label, rows in sample_specs:
        for position, (_, row) in enumerate(rows.iterrows(), start=1):
            choices.append(
                {
                    "key": f"{group}:{position}",
                    "group": group,
                    "label": f"{label} {position}",
                    "amount": float(row["Amount"]),
                    "actual_class": int(row["Class"]),
                    "features": {name: float(row[name]) for name in FEATURE_ORDER},
                }
            )
    return choices


def get_sample_features(sample_key):
    if not sample_key:
        return None
    for choice in get_sample_choices():
        if choice["key"] == sample_key:
            return choice
    return None


def format_transaction_row(row):
    """
    Ensures all new keys are populated even for legacy records in the database.
    """
    if not row:
        return None
    row_dict = dict(row)
    prediction = row_dict.get("prediction", 0)
    probability = row_dict.get("probability", 0.0)
    timestamp = row_dict.get("timestamp", "")

    if not row_dict.get("reference_id"):
        date_part = "20260712"
        if timestamp:
            try:
                date_part = timestamp.split("T")[0].replace("-", "")
            except Exception:
                pass
        prefix = "FRD" if prediction else "TXN"
        row_dict["reference_id"] = f"{prefix}-{date_part}-OLD{row_dict.get('id', 0):03d}"

    if not row_dict.get("risk_level"):
        row_dict["risk_level"] = risk_category(probability)

    if not row_dict.get("status"):
        row_dict["status"] = "Blocked" if prediction else "Approved"

    if not row_dict.get("action_taken"):
        row_dict["action_taken"] = "Transaction temporarily blocked" if prediction else "Transaction approved"

    if not row_dict.get("created_date") or not row_dict.get("created_time"):
        if timestamp:
            try:
                parts = timestamp.split("T")
                row_dict["created_date"] = parts[0]
                if len(parts) > 1:
                    row_dict["created_time"] = parts[1].split(".")[0].split("+")[0]
                else:
                    row_dict["created_time"] = "00:00:00"
            except Exception:
                row_dict["created_date"] = "N/A"
                row_dict["created_time"] = "N/A"
    if row_dict.get("amount") is None:
        row_dict["amount"] = 120.00

    return row_dict


def ensure_transaction_table():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction INTEGER,
                probability REAL,
                timestamp TEXT
            )
            """
        )
        conn.commit()

        # Dynamically upgrade schema for legacy databases
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(transactions)")
        columns = [col[1] for col in cursor.fetchall()]

        new_columns = {
            "reference_id": "TEXT",
            "risk_level": "TEXT",
            "action_taken": "TEXT",
            "status": "TEXT",
            "created_date": "TEXT",
            "created_time": "TEXT",
            "amount": "REAL"
        }

        for col_name, col_type in new_columns.items():
            if col_name not in columns:
                conn.execute(f"ALTER TABLE transactions ADD COLUMN {col_name} {col_type}")
        conn.commit()


def save_transaction(prediction, probability, reference_id=None, risk_level=None, action_taken=None, status=None, created_date=None, created_time=None, amount=None):
    ensure_transaction_table()
    now = timezone.localtime()
    if not created_date:
        created_date = now.strftime("%Y-%m-%d")
    if not created_time:
        created_time = now.strftime("%H:%M:%S")
    if not reference_id:
        date_str = now.strftime("%Y%m%d")
        import random, string
        random_suffix = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        prefix = "FRD" if prediction else "TXN"
        reference_id = f"{prefix}-{date_str}-{random_suffix}"
    if not risk_level:
        risk_level = risk_category(probability)
    if not status:
        status = "Blocked" if prediction else "Approved"
    if not action_taken:
        action_taken = "Transaction temporarily blocked" if prediction else "Transaction approved"

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO transactions (
                prediction, probability, timestamp, reference_id, risk_level, action_taken, status, created_date, created_time, amount
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                prediction,
                probability,
                now.isoformat(),
                reference_id,
                risk_level,
                action_taken,
                status,
                created_date,
                created_time,
                amount
            ),
        )
        conn.commit()
        return cursor.lastrowid


def get_recent_transactions(limit=8, prediction_filter="all"):
    ensure_transaction_table()
    where_clause = ""
    params = []
    if prediction_filter == "fraud":
        where_clause = "WHERE prediction = ?"
        params.append(1)
    elif prediction_filter == "legitimate":
        where_clause = "WHERE prediction = ?"
        params.append(0)
    params.append(limit)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"""
            SELECT id, prediction, probability, timestamp, reference_id, risk_level, action_taken, status, created_date, created_time, amount
            FROM transactions
            {where_clause}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
    return [format_transaction_row(row) for row in rows]


def get_transaction_by_id(tx_id):
    ensure_transaction_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, prediction, probability, timestamp, reference_id, risk_level, action_taken, status, created_date, created_time, amount
            FROM transactions
            WHERE id = ?
            """,
            (tx_id,),
        ).fetchone()
    return format_transaction_row(row) if row else None


def get_transaction_by_ref_id(reference_id):
    ensure_transaction_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, prediction, probability, timestamp, reference_id, risk_level, action_taken, status, created_date, created_time, amount
            FROM transactions
            WHERE reference_id = ?
            """,
            (reference_id,),
        ).fetchone()
    return format_transaction_row(row) if row else None


def get_prediction_summary():
    ensure_transaction_table()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        metrics_row = conn.execute(
            """
            SELECT 
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) AS fraud_predictions,
                SUM(CASE WHEN prediction = 0 THEN 1 ELSE 0 END) AS legit_predictions,
                AVG(probability) AS avg_risk_score
            FROM transactions
            """
        ).fetchone()
        
        total = metrics_row["total_predictions"] or 0
        fraud = metrics_row["fraud_predictions"] or 0
        legit = metrics_row["legit_predictions"] or 0
        avg_risk = metrics_row["avg_risk_score"] or 0.0
        fraud_rate = fraud / total if total > 0 else 0.0
        
        latest_fraud_row = conn.execute(
            """
            SELECT id, prediction, probability, timestamp, reference_id, risk_level, status, action_taken, created_date, created_time, amount
            FROM transactions
            WHERE prediction = 1
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        
        latest_legit_row = conn.execute(
            """
            SELECT id, prediction, probability, timestamp, reference_id, risk_level, status, action_taken, created_date, created_time, amount
            FROM transactions
            WHERE prediction = 0
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        
        latest_fraud = format_transaction_row(latest_fraud_row) if latest_fraud_row else None
        latest_legit = format_transaction_row(latest_legit_row) if latest_legit_row else None
        
        return {
            "total_predictions": total,
            "fraud_predictions": fraud,
            "legit_predictions": legit,
            "fraud_rate": fraud_rate,
            "avg_risk_score": avg_risk,
            "latest_fraud": latest_fraud,
            "latest_legit": latest_legit,
        }


def get_prediction_counts():
    ensure_transaction_table()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT prediction, COUNT(*) AS total
            FROM transactions
            GROUP BY prediction
            """
        ).fetchall()
    counts = {0: 0, 1: 0}
    counts.update({int(prediction): int(total) for prediction, total in rows})
    return counts


def load_dashboard_metrics(filters=None):
    pandas = _import_or_raise("pandas")
    filters = filters or {}
    usecols = ["Time", "Amount", "Class"]
    df = load_transaction_dataset(usecols=usecols)

    amount_min = _safe_float(filters.get("amount_min"), None)
    amount_max = _safe_float(filters.get("amount_max"), None)
    class_filter = filters.get("class_filter", "all")

    if amount_min is not None:
        df = df[df["Amount"] >= amount_min]
    if amount_max is not None:
        df = df[df["Amount"] <= amount_max]
    if class_filter == "fraud":
        df = df[df["Class"] == 1]
    elif class_filter == "legitimate":
        df = df[df["Class"] == 0]

    total = int(len(df))
    fraud = int(df["Class"].sum())
    legitimate = total - fraud
    fraud_rate = fraud / total if total else 0
    fraud_amount = float(df.loc[df["Class"] == 1, "Amount"].sum())
    avg_amount = float(df["Amount"].mean())

    amount_bins = pandas.cut(
        df["Amount"],
        bins=[-0.01, 10, 50, 100, 250, 500, 1000, float("inf")],
        labels=["0-10", "10-50", "50-100", "100-250", "250-500", "500-1k", "1k+"],
    )
    amount_distribution = [
        {"label": str(label), "count": int(count)}
        for label, count in amount_bins.value_counts(sort=False).items()
    ]

    class_distribution = [
        {"label": "Legitimate", "count": legitimate},
        {"label": "Fraud", "count": fraud},
    ]

    hour_series = (df["Time"] // 3600).astype(int) % 24
    fraud_by_hour = (
        df.loc[df["Class"] == 1]
        .assign(hour=hour_series[df["Class"] == 1])
        .groupby("hour")
        .size()
        .reindex(range(24), fill_value=0)
    )
    hourly_fraud = [{"hour": int(hour), "count": int(count)} for hour, count in fraud_by_hour.items()]

    time_bins = pandas.cut(df["Time"], bins=12, duplicates="drop")
    transaction_trend = [
        {"label": f"T{index + 1}", "count": int(count)}
        for index, count in enumerate(time_bins.value_counts(sort=False).tolist())
    ]

    top_transactions = (
        df.sort_values("Amount", ascending=False)
        .head(8)[["Time", "Amount", "Class"]]
        .to_dict("records")
    )

    prediction_counts = get_prediction_counts()

    return {
        "total": total,
        "fraud": fraud,
        "legitimate": legitimate,
        "fraud_rate": fraud_rate,
        "fraud_amount": fraud_amount,
        "avg_amount": avg_amount,
        "class_distribution": class_distribution,
        "amount_distribution": amount_distribution,
        "hourly_fraud": hourly_fraud,
        "transaction_trend": transaction_trend,
        "top_transactions": top_transactions,
        "prediction_counts": prediction_counts,
        "filters": {
            "amount_min": "" if amount_min is None else amount_min,
            "amount_max": "" if amount_max is None else amount_max,
            "class_filter": class_filter,
        },
        "class_options": CLASS_OPTIONS,
    }


def _safe_float(value, default):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_plot_path(name):
    allowed = {
        "confusion_matrix": "confusion_matrix.png",
        "roc_curve": "roc_curve.png",
        "precision_recall": "precision_recall.png",
    }
    filename = allowed.get(name)
    if not filename:
        return None
    path = settings.BASE_DIR / "artifacts" / "plots" / filename
    return path if path.exists() else None


def get_assistant_answer(question):
    cleaned = (question or "").strip()
    if not cleaned:
        return "Welcome to FraudDesk AI Knowledge Assistant. Please enter a question or click one of the quick topics below."
    
    q_lower = cleaned.lower()

    # Category 1: Project Overview
    if any(k in q_lower for k in ["overview", "project", "workflow", "objective", "architecture", "technology", "technologies"]):
        return (
            "### Project Overview & System Workflow\n\n"
            "The FraudDesk system is an enterprise-grade Credit Card Fraud Detection platform designed to monitor transaction flows, "
            "analyze potential risks using Machine Learning, and coordinate banking security actions.\n\n"
            "**Key Components:**\n"
            "*   **Objective:** Detect fraudulent credit card transactions in real-time before settlement completes.\n"
            "*   **Workflow:** Transactions are captured by the gateway, feature extraction formats numerical attributes, "
            "the offline XGBoost classifier scores the transaction risk, and positive signals initiate a response workflow.\n"
            "*   **Architecture:** Built on a standard Django model-view-template structure serving offline pickled ML models, keeping operations fully local and secure."
        )

    # Category 2: Dashboard
    elif any(k in q_lower for k in ["dashboard", "chart", "statistics", "card", "rate"]):
        return (
            "### Dashboard Guide & Analytics\n\n"
            "The dashboard serves as the primary console for security analysts, displaying aggregated transaction statistics and historical model activities.\n\n"
            "**Key Metrics Displayed:**\n"
            "*   **KPI Panels:** Monitor total predictions, identified fraud cases, clean approvals, average transaction value, and real-time fraud rates.\n"
            "*   **Class Balance Chart:** Visual representation of fraudulent vs legitimate transactions.\n"
            "*   **Trend Charts:** Plots transaction volume over time and fraud occurrence frequencies per hour.\n"
            "*   **Model Performance:** Displays pre-computed evaluation matrices (Confusion Matrix, ROC Curve, and Precision-Recall Curve) stored as static artifacts."
        )

    # Category 3: Prediction
    elif any(k in q_lower for k in ["prediction", "how fraud is detected", "confidence", "probability", "risk level", "evaluation"]):
        return (
            "### Live Prediction & Scoring Process\n\n"
            "The prediction module simulates real-time transaction ingestion and scoring.\n\n"
            "**Redirection Flow:**\n"
            "*   **Inputs:** Ingests transaction metadata including Time, Amount, and PCA-anonymized features V1 to V28.\n"
            "*   **Standardization:** Input features are pre-processed and transformed using a saved `StandardScaler` template.\n"
            "*   **Classification:** The processed record is scored by the trained XGBoost model, returning a binary classification and probability score.\n"
            "*   **Evaluation:** Automatically redirects to the dedicated evaluation page to present the full risk report."
        )

    # Category 4: Dataset
    elif any(k in q_lower for k in ["v1", "v2", "v28", "amount", "time", "anonymi", "class", "dataset"]):
        return (
            "### Credit Card Dataset Features\n\n"
            "The system is evaluated on a historical European cardholders dataset containing transactions.\n\n"
            "**Anonymization & Fields:**\n"
            "*   **V1 - V28:** Numerical features obtained via Principal Component Analysis (PCA) transformation to protect customer confidentiality and card information.\n"
            "*   **Time:** Elapsed seconds between the current transaction and the first transaction in the dataset.\n"
            "*   **Amount:** The transaction value.\n"
            "*   **Class:** The output label where `0` denotes legitimate and `1` denotes fraud."
        )

    # Category 5: Model
    elif any(k in q_lower for k in ["algorithm", "random forest", "trained", "preprocess", "scale", "split"]):
        return (
            "### Machine Learning Model Details\n\n"
            "The platform employs an offline XGBoost Classifier chosen for its speed and classification performance.\n\n"
            "**Training Workflow:**\n"
            "*   **Class Imbalance:** Solved using Synthetic Minority Over-sampling Technique (SMOTE) to prevent prediction bias.\n"
            "*   **Train-Test Split:** Splitting dataset into training (80%) and testing (20%) segments.\n"
            "*   **Preprocessing:** Standardizing features using Scikit-Learn's `StandardScaler`.\n"
            "*   **Serialization:** Preserving model artifacts as pickle files (`.pkl`) for offline predictions."
        )

    # Category 6: Performance
    elif any(k in q_lower for k in ["accuracy", "precision", "recall", "f1", "roc", "confusion", "pr-curve"]):
        return (
            "### Model Performance Metrics\n\n"
            "Model evaluation metrics are computed to ensure system reliability and low false alarm rates.\n\n"
            "**Core Metrics:**\n"
            "*   **Accuracy:** The ratio of correct predictions.\n"
            "*   **Precision:** The proportion of flagged transactions that are actually fraudulent.\n"
            "*   **Recall (Sensitivity):** The proportion of actual fraud cases successfully flagged by the model.\n"
            "*   **F1-Score:** The harmonic mean of Precision and Recall.\n"
            "*   **ROC & PR Curves:** Graphically represent threshold balances to minimize customer friction."
        )

    # Category 7: Fraud Response
    elif any(k in q_lower for k in ["block", "notification", "security alert", "investigation", "risk assessment", "recommendation"]):
        return (
            "### Enterprise Fraud Response Workflow\n\n"
            "Once a transaction is flagged as fraudulent, the platform triggers a simulated banking response center.\n\n"
            "**Triggered Events:**\n"
            "*   **Transaction Blocked:** Funds transfer hold placed immediately.\n"
            "*   **Customer Alert:** Verification message dispatched to cardholder.\n"
            "*   **Security Alert:** Alert routed to the Fraud Investigation Team.\n"
            "*   **Status Code:** Classified as High Risk with status set to 'Blocked'.\n"
            "*   **Recommendations:** Suggests card freezing, customer contact, and PIN resets."
        )

    # Category 8: Reports
    elif any(k in q_lower for k in ["report", "history", "reference id", "download"]):
        return (
            "### Audit Trail & Printable Reports\n\n"
            "The system generates audit reports for every transaction scored, facilitating bank compliance checks.\n\n"
            "**Key Elements:**\n"
            "*   **Prediction History:** Stores Reference ID, Risk Level, Status, Action Taken, and timestamps in SQLite.\n"
            "*   **Reference ID:** Structured identifiers using `FRD-YYYYMMDD-XXXXXX` (Fraud) and `TXN-YYYYMMDD-XXXXXX` (Legitimate).\n"
            "*   **Printable Report:** Page configured with CSS `@media print` directives for clean printing or saving to PDF."
        )

    # Category 9: System
    elif any(k in q_lower for k in ["folder", "django", "ml pipeline", "deploy", "auth"]):
        return (
            "### System Architecture & Organization\n\n"
            "FraudDesk is built using a clean folder structure separating Django web routing from ML pipeline components.\n\n"
            "**Layout:**\n"
            "*   `fraud_project/`: Django configuration, settings, and main URL entrypoints.\n"
            "*   `fraud_app/`: Routing views, templates, static assets, and SQLite DB queries.\n"
            "*   `src/`: Core Python pipeline, SMOTE balancing, XGBoost training modules.\n"
            "*   `data/` & `models/`: Directories containing dataset files and serialized pickle binaries."
        )

    # Category 10: Risk Assessment
    elif any(k in q_lower for k in ["risk"]):
        return (
            "### Risk Assessment Thresholds\n\n"
            "Transactions are grouped into risk categories based on their computed fraud probabilities:\n"
            "*   **High Risk:** Probabilities >= 75%. Triggers card block and immediate security alert.\n"
            "*   **Medium Risk:** Probabilities between 35% and 75%. Triggers analyst review.\n"
            "*   **Low Risk:** Probabilities < 35%. Marked as legitimate and approved."
        )

    # Category 11: FAQ
    elif any(k in q_lower for k in ["faq", "frequently", "question"]):
        return (
            "### Frequently Asked Questions\n\n"
            "*   **Is the system real?** No, banking activities, SMS, and alert notifications are simulated offline for security.\n"
            "*   **Can the model run without internet?** Yes, the XGBoost classifier is stored locally, operating fully offline.\n"
            "*   **How is a customer notified?** A simulation alert box displays the generated warning message in the console."
        )

    return (
        "I can only explain the Credit Card Fraud Detection System, its dashboard, machine learning model, "
        "prediction process, reports, dataset, and fraud response workflow."
    )
