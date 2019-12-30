import dtlpy as dl
import logging

logger = logging.getLogger(name=__name__)
from importlib import import_module


def pred_run(checkpoint_path, name, home_path, progress=None):
    predict = getattr(import_module('.adapter', 'zoo.' + name), 'predict')

    return predict(home_path, checkpoint_path=checkpoint_path)
