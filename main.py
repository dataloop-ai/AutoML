import dtlpy as dl
import logging

logger = logging.getLogger(name=__name__)
from tensorflow import keras
from launch_pad import Experiment


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

        (x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
        x = x.astype('float32') / 255.
        val_x = val_x.astype('float32') / 255.

        x = x[:10000]
        y = y[:10000]
        model = model['model_str']

        experiment = Experiment(hp_values, model, configs, x, y)
        metrics = experiment.run()
        logging.info('return value :', metrics)
        return metrics

if __name__ == "__main__":
    """
    Run this main to locally debug your plugin
    """
    dl.plugins.test_local_plugin()
