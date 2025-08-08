# app_logger.py

import logging

class AppLogger:
    def __init__(self, logger_name: str, level=logging.DEBUG):
        self.logger_name = logger_name
        self.level = level

    def setupLogger(self):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.level)

        # Avoid duplicate handlers if logger is called multiple times
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(self.level)

            logger.addHandler(console_handler)

        return logger
