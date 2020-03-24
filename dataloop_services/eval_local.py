import dtlpy as dl
import logging
from logging_utils import logginger
logger = logging.getLogger(name=__name__)
from importlib import import_module


class LocalEvalConnector():

    def __init__(self):
        self.logger = logginger(__name__)

    def run(self, devices, model_specs, hp_values):
        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')

        adapter = cls(devices, model_specs, hp_values)

        adapter.eval()
