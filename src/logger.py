import os
import logging
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure the logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("logs/application.log", maxBytes=5*1024*1024, backupCount=3),  # Log rotation
        logging.StreamHandler()  # Log to console as well
    ]
)

# Function to get the logger
def get_logger(name):
    return logging.getLogger(name)
