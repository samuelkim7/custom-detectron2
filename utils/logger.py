import os
import logging


def custom_logger(name, log_file_path) -> logging.Logger:
    """
    custom logger for printing training configurations
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Printing formatter
    formatter = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if not os.path.isdir(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger