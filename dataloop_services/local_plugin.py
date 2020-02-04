import dtlpy as dl
import logging

logger = logging.getLogger(name=__name__)
from importlib import import_module


class LocalTrialConnector():

    def __init__(self, service_name):
        self.service_name = service_name

    def run(self, devices, model_specs, hp_values):
        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')

        final = 1 if self.service_name == 'trainer' else 0
        adapter = cls(devices, model_specs, hp_values, final)
        if hasattr(adapter, 'reformat'):
            adapter.reformat()
        if hasattr(adapter, 'data_loader'):
            adapter.data_loader()
        if hasattr(adapter, 'preprocess'):
            adapter.preprocess()
        if hasattr(adapter, 'build'):
            adapter.build()
        adapter.train()

        if final:
            return adapter.get_checkpoint()
        else:
            metrics = adapter.get_metrics()
            if type(metrics) is not dict:
                raise Exception('adapter, get_metrics method must return dict object')
            if type(metrics['val_accuracy']) is not float:
                raise Exception(
                    'adapter, get_metrics method must return dict with only python floats. '
                    'Not numpy floats or any other objects like that')
            return metrics
