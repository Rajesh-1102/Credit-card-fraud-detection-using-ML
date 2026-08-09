import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.pipeline import run_pipeline
from src.database import create_table

if __name__ == "__main__":
    create_table()
    run_pipeline()

