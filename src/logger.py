import logging
import os
from datetime import datetime

# Define the log file name with the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%Y_H_%M_%S')}.log"

# Define the log path
log_path = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it does not exist
os.makedirs(log_path, exist_ok=True)

# Complete log file path
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format for log messages
)

# Example of logging an information message
if __name__ == "__main__":
    logging.info("Logging has been set up successfully.")
