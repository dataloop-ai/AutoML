import dtlpy as dl
import logging

logger = logging.getLogger(name=__name__)
from importlib import import_module


def pred_run(checkpoint_path, name, data, progress=None):
    predict = getattr(import_module('.adapter', 'zazoo.' + name), 'predict')

    return predict(data, checkpoint_path=checkpoint_path)
