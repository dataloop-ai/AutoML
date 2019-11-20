import dtlpy as dl
import logging
# from retinanet import HyperModel
from keras_toy_model import HyperModel
logger = logging.getLogger(name=__name__)



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

    def run(self, model, hp_values, progress=None):

        h_model = HyperModel(model, hp_values)
        h_model.data_loader()
        h_model.train()

        # logging.info('return value :', metrics)
        # return metrics


if __name__ == "__main__":
    """
    Run this main to locally debug your plugin
    """
    dl.plugins.test_local_plugin()
