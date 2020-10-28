import logging
import sys
import os

LOGGING_FORMAT = '[%(levelname)s] %(asctime)s %(name)s : %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def init_logging(module_name, filename='logger.conf'):
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
    logger = logging.getLogger(module_name)

    if os.path.exists(filename):
        os.remove(filename)

    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(logging.Formatter(fmt=LOGGING_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(fileHandler)

    return logger

def logginger(module_name, filename='logger.conf'):
    logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
    logger = logging.getLogger(module_name)

    if len(logger.handlers) < 1:
        fileHandler = logging.FileHandler(filename)
        fileHandler.setFormatter(logging.Formatter(fmt=LOGGING_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(fileHandler)
        
    return logger


