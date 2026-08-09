import os
os.makedirs("artifacts/reports", exist_ok=True)
import sqlite3
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def export_to_csv():
    logger.info("Exporting data for Power BI...")

    conn = sqlite3.connect("database/fraud.db")

    df = pd.read_sql("SELECT * FROM transactions", conn)

    df.to_csv("artifacts/reports/fraud_data.csv", index=False)

    conn.close()

    logger.info("Data exported successfully!")


if __name__ == "__main__":
    export_to_csv()