# FraudDesk - Enterprise Credit Card Fraud Detection System

FraudDesk is an enterprise-grade risk intelligence console that integrates an offline Machine Learning pipeline with a Django-powered banking panel. The system is designed to ingestion transaction data, scale features, predict fraud risks in real-time, coordinate automated compliance workflows, and log detailed audit timelines.

---

## 🛡️ Core Features

### 1. Risk Monitoring Dashboard
*   **KPI Metrics:** Real-time summary panels tracking total predictions, flagged fraud, approvals, average transaction size, and aggregate fraud rates.
*   **Case Feeds:** Separate live feeds displaying the latest suspicious (blocked) and legitimate (cleared) transactions.
*   **Performance Metrics:** Embedded graphs showcasing model characteristics (Confusion Matrix, ROC Curve, and Precision-Recall Curve).

### 2. Dedicated Risk Evaluation Page
*   **Scoring Breakdown:** Interactive summaries showing probabilities, standard deviations, PCA features, and processing durations.
*   **AI Decision Explanation:** Natural language descriptions explaining *why* the XGBoost classifier flagged or cleared the transaction.
*   **Action Desk:** Real-time indicators showing account holds, security ticket routing, and customer communication flags.

### 3. Automated Fraud Response Center
*   **Reference ID Generator:** Unique IDs generated dynamically:
    *   `FRD-YYYYMMDD-XXXXXX` for suspicious/fraudulent cases.
    *   `TXN-YYYYMMDD-XXXXXX` for legitimate cases.
*   **Audit Timelines:** Trace logs recording each step (Received $\rightarrow$ Extracted $\rightarrow$ Analyzed $\rightarrow$ Scored $\rightarrow$ Blocked/Approved $\rightarrow$ Alerted).
*   **Recommended Actions:** Predefined follow-up protocols for bank operators (freeze cards, verify identities, reset PINs).

### 4. FraudDesk AI Knowledge Assistant
*   **Offline Support:** Fully local query assistant that maps search keywords to internal system documentation.
*   **Topic Coverage:** Answers questions about folder structures, ML training parameters, dataset anonymity, metric evaluations, and response checklists.

### 5. 3-Way Cycle Theme Switcher
*   **Modes:** Cycle through **Day Mode (Sun)**, **Night Mode (Moon)**, and **System Default (Monitor)**.
*   **Circular Layout:** Simple, clean icon-only toggle button with active OS preference synchronization.

---

## ⚙️ Tech Stack

*   **Core Backend:** Python, Django 5.2+
*   **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
*   **Data Processing:** Pandas, NumPy
*   **Visualizations:** Matplotlib, Seaborn
*   **Database:** SQLite3

---

## 📂 Project Structure

```text
Credit-card-fraud-detection-main/
│
├── data/                       # Dataset directories (raw & processed)
├── database/                   # SQLite database location (fraud.db)
├── fraud_app/                  # Django application (views, routing, models)
├── fraud_project/              # Django project configurations & settings
├── models/                     # Pickled model files (XGBoost classifier & scaler)
├── src/                        # Machine Learning pipeline source code
│   ├── components/             # Preprocessing, training, and evaluation scripts
│   ├── database.py             # SQLite helper and table migration scripts
│   ├── pipeline.py             # Pipeline execution orchestration
│   └── predict.py              # Serialized prediction wrapper
│
├── static/                     # CSS stylesheets and theme assets
├── templates/                  # Django HTML template directory
│
├── main.py                     # ML pipeline entrypoint
├── manage.py                   # Django management script
├── requirements.txt            # Python dependencies
└── README.md                   # Consolidated project documentation
```

---

## 🚀 Installation & Local Setup

### 1. Activate Environment
Clone this repository to your workspace, create a virtual environment, and activate it:
```powershell
python -m venv venv
venv\Scripts\activate
```

### 2. Install Packages
Install the required packages:
```powershell
pip install -r requirements.txt
```

### 3. Run Database Migrations
Initialize the SQLite database schema and migrate models:
```powershell
python manage.py migrate
```

### 4. Boot Django Development Server
Start the server locally:
```powershell
python manage.py runserver
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser. 

*Note: You can log in using the pre-configured local administrator credentials:*
*   **Username:** `admin`
*   **Password:** `admin123`

---

## 🤖 Model Training & ML Pipeline

The system is trained on a credit card transaction dataset containing PCA-anonymized features (V1 to V28), Time, and Amount.

To preprocess data, handle class imbalances (using SMOTE), scale variables, train the XGBoost classifier, and generate performance graphs, run:
```powershell
python main.py
```
This script will output the updated model binaries and metric curves:
*   `models/fraud_model.pkl` (Best model artifact)
*   `models/scaler.pkl` (StandardScaler configuration)
*   `artifacts/plots/` (Confusion matrix, ROC, and PR curves)

---

## 🔌 API Documentation

FraudDesk exposes a local JSON prediction API endpoint for high-frequency automated scoring.

*   **Endpoint:** `POST http://127.0.0.1:8000/api/predict/`
*   **Headers:** `Content-Type: application/json`

### JSON Request Payload
```json
{
  "Time": 1282,
  "V1": -1.25,
  "V2": 0.45,
  "V3": 2.1,
  "V4": -0.85,
  "V5": 0.12,
  "V6": -0.34,
  "V7": 0.78,
  "V8": -0.09,
  "V9": 0.56,
  "V10": -0.23,
  "V11": 1.05,
  "V12": -0.67,
  "V13": 0.11,
  "V14": -0.89,
  "V15": 0.45,
  "V16": -0.12,
  "V17": 0.34,
  "V18": -0.56,
  "V19": 0.89,
  "V20": -0.05,
  "V21": -0.15,
  "V22": 0.35,
  "V23": -0.18,
  "V24": 0.42,
  "V25": -0.07,
  "V26": 0.13,
  "V27": -0.22,
  "V28": 0.08,
  "Amount": 79.99
}
```

### JSON Response Payload
```json
{
  "prediction": 0,
  "probability": 0.012456,
  "label": "Legitimate",
  "reference_id": "TXN-20260712-A8D9E3",
  "risk_category": "Low Risk",
  "status": "Approved",
  "action_taken": "Transaction approved",
  "created_date": "2026-07-12",
  "created_time": "22:45:00"
}
```

---

## 📝 Compliance & Audit Guidelines
All predictions, probability calculations, status shifts, and action logs are captured in `database/fraud.db` inside the `transactions` table. Bank supervisors can click **"Download Report"** on any transaction evaluation screen to print or save a formal compliance statement (CSS print-optimized format).
