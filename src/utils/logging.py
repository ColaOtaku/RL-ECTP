import os
import sys
import logging

def get_logger(log_dir, name, log_filename, level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        # File handler
        file_formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        print('Log directory:', log_dir)
    
    return logger