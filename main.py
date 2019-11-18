import dtlpy as dl
import logging
from my_model import Model

logger = logging.getLogger(name=__name__)
from keras_toy_model import HyperModel


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

    def run(self, model, configs, hp_values, progress=None):
        model = model['model_str']

        model = Model(model)
        model.data_loader(configs)
        model.add_preprocess(hp_values)
        model.build(hp_values)
        metrics = model.train()

        logging.info('return value :', metrics)
        return metrics


if __name__ == "__main__":
    """
    Run this main to locally debug your plugin
    """
    dl.plugins.test_local_plugin()
