import os
import sqlite3
from datetime import datetime

from src.logger import get_logger

logger = get_logger(__name__)

os.makedirs("database", exist_ok=True)

DB_PATH = "database/fraud.db"


def create_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn


def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction INTEGER,
            probability REAL,
            timestamp TEXT
        )
    """)
    conn.commit()

    # Dynamically upgrade schema for legacy databases
    cursor.execute("PRAGMA table_info(transactions)")
    columns = [col[1] for col in cursor.fetchall()]

    new_columns = {
        "reference_id": "TEXT",
        "risk_level": "TEXT",
        "action_taken": "TEXT",
        "status": "TEXT",
        "created_date": "TEXT",
        "created_time": "TEXT"
    }

    for col_name, col_type in new_columns.items():
        if col_name not in columns:
            cursor.execute(f"ALTER TABLE transactions ADD COLUMN {col_name} {col_type}")
    conn.commit()
    conn.close()

    logger.info("Database table ready.")


def insert_transaction(prediction, probability):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO transactions (prediction, probability, timestamp)
        VALUES (?, ?, ?)
    """, (prediction, probability, datetime.now().isoformat()))

    conn.commit()
    conn.close()

    logger.info("Transaction saved to database.")