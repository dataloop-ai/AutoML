import logging
import sys
import os


def init_logging(module_name, filename='logger.conf'):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(module_name)

    if os.path.exists(filename):
        #os.remove(filename)
        pass
    fileHandler = logging.FileHandler(filename)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    logger.addHandler(consoleHandler)

    return logger

def logginger(module_name, filename='logger.conf'):
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logger = logging.getLogger(module_name)

    fileHandler = logging.FileHandler(filename)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    logger.addHandler(consoleHandler)

    return logger


