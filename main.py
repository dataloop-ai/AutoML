import logging

logger = logging.getLogger(name=__name__)
from importlib import import_module

class PluginRunner(dl.BasePluginRunner):
    """
    Plugin runner class

    """

    def __init__(self, **kwargs):
        """
        Init plugin attributes here
        
        :param kwargs: config params
        :return:
        """

    def run(self, model_specs, hp_values, progress=None):
        cls = getattr(import_module('.' + 'adapter', model_specs['name']), 'AdapterModel')
        adapter = cls(model_specs, hp_values)
        if hasattr(adapter, 'reformat'):
            adapter.reformat()
        if hasattr(adapter, 'data_loader'):
            adapter.data_loader()
        if hasattr(adapter, 'preprocess'):
            adapter.preprocess()
        if hasattr(adapter, 'build'):
            adapter.build()
        adapter.train()

        metrics = adapter.get_metrics()

        if type(metrics) is not dict:
            raise Exception('adapter, get_metrics method must return dict object')
        if type(metrics['val_accuracy']) is not float:
            raise Exception(
                'adapter, get_metrics method must return dict with only python floats. '
                'Not numpy floats or any other objects like that')

        return metrics
