import dtlpy as dl
import logging
from logging_utils import logginger
logger = logging.getLogger(name=__name__)
from importlib import import_module


class ServiceRunner(dl.BaseServiceRunner):
    """
    Plugin runner class

    """
    def __init__(self):
        self.logger = logginger(__name__)

    def run(self, checkpoint_path, model_specs):
        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')

        home_path = model_specs['data']['home_path']

        adapter = cls()
        adapter.predict(home_path=home_path, checkpoint_path=checkpoint_path)
