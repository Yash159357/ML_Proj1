import logging
import os
import sys
from datetime import datetime

def setup_logging():
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # Create folder for this run
    log_dir = os.path.join(os.getcwd(), "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Create a log file inside that folder with the same timestamp
    log_file = os.path.join(log_dir, f"{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file
