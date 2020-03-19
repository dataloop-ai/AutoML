import dtlpy as dl
import logging
from logging_utils import logginger
logger = logging.getLogger(name=__name__)
from importlib import import_module


class LocalTrialConnector():

    def __init__(self, service_name):
        self.service_name = service_name
        self.logger = logginger(__name__)

    def run(self, devices, model_specs, hp_values):
        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')

        final = 1 if self.service_name == 'trainer' else 0
        adapter = cls(devices, model_specs, hp_values)
        if hasattr(adapter, 'reformat'):
            adapter.reformat()
        if hasattr(adapter, 'data_loader'):
            adapter.data_loader()
        if hasattr(adapter, 'preprocess'):
            adapter.preprocess()
        if hasattr(adapter, 'build'):
            adapter.build()
        self.logger.info('commencing training . . . ')
        adapter.train()
        self.logger.info('training finished')
        if final:
            return adapter.get_checkpoint()
        else:
            metrics_and_checkpoint_dict = adapter.get_metrics_and_checkpoint()
            if type(metrics_and_checkpoint_dict) is not dict:
                raise Exception('adapter, get_metrics method must return dict object')
            if type(metrics_and_checkpoint_dict['metrics']['val_accuracy']) is not float:
                raise Exception(
                    'adapter, get_metrics method must return dict with only python floats. '
                    'Not numpy floats or any other objects like that')
            return metrics_and_checkpoint_dict
