import logging
import os
import sys
from datetime import datetime

# Single folder for all logs
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# One file per run
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
LOG_FILE = os.path.join(LOG_DIR, f"log_{timestamp}.log")

# Configure logger once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("ml_project_logger")
logger.info(f"Logging initialized. Log file: {LOG_FILE}")
