import logging
from pathlib import Path
from src.config import LOG_DIR

LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / "project.log"

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"   # optional but recommended
)

def get_logger(name):
    return logging.getLogger(name)