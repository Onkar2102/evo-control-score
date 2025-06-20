import os
import logging
from logging.handlers import RotatingFileHandler
import platform
import getpass
import datetime
import json

# Global variable to store the log file path for the current run
_CURRENT_LOG_FILE = None

def get_run_id():
    log_index_file = "logs/log_index.json"
    today = datetime.datetime.now().strftime("%Y%m%d")

    if not os.path.exists("logs"):
        os.makedirs("logs")

    if os.path.exists(log_index_file):
        with open(log_index_file, "r") as f:
            log_data = json.load(f)
    else:
        log_data = {}

    today_run_id = log_data.get(today, 0) + 1
    log_data[today] = today_run_id

    with open(log_index_file, "w") as f:
        json.dump(log_data, f)

    return today_run_id

def get_log_filename():
    global _CURRENT_LOG_FILE
    
    # If we already have a log file for this run, return it
    if _CURRENT_LOG_FILE is not None:
        return _CURRENT_LOG_FILE
    
    # Create a new log file only once per run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = get_run_id()
    user = getpass.getuser()
    machine = platform.node().split('.')[0]
    device_info = f"{user}@{machine}".replace(" ", "_")

    _CURRENT_LOG_FILE = f"logs/{timestamp}_run{run_id}_{device_info}.log"
    return _CURRENT_LOG_FILE

def get_logger(name: str = "default_logger", log_file: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Stream handler for console output
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if not log_file:
            log_file = get_log_filename()
        file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.setLevel(log_level)
        logger.debug("Logger initialized for %s", name)
    return logger