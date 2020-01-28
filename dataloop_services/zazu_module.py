import logging
import os
import torch
import json
import dtlpy as dl
from importlib import import_module
from plugin_utils import maybe_download_data
logger = logging.getLogger(__name__)


class ServiceRunner(dl.BaseServiceRunner):
    """
    Plugin runner class

    """

    def __init__(self, package_name):
        self.package_name = package_name
        self.path_to_best_checkpoint = 'checkpoint.pt'
        self.path_to_metrics = 'metrics.json'
        self.path_to_tensorboard_dir = 'runs'
        logger.info(self.package_name + ' initialized')

    def search(self, dataset, model_specs, hp_values, configs=None, progress=None):
        configs_path = os.path.join(this_path, 'configs.json')
        configs = ConfigSpec(configs_path)
        opt_model = OptModel()
        opt_model.add_child_spec(configs, 'configs')
        zazu = ZaZu(opt_model, remote=True)
        zazu.find_best_model()
        zazu.hp_search()
    def train(self):
        configs_path = os.path.join(this_path, 'configs.json')
        configs = ConfigSpec(configs_path)
        opt_model = OptModel()
        opt_model.add_child_spec(configs, 'configs')
        zazu = ZaZu(opt_model, remote=True)
        zazu.train_new_model()
    def predict():
        configs_path = os.path.join(this_path, 'configs.json')
        configs = ConfigSpec(configs_path)
        opt_model = OptModel()
        opt_model.add_child_spec(configs, 'configs')
        zazu = ZaZu(opt_model, remote=True)
        zazu.run_inference()

if __name__ == "__main__":
    import dtlpy as dl
    zazu_service = dl.services.get('Zazu')
    zazu_service.invoke(execution_input=[dl.FunctionIO('Dataset', value='IPM'),
                                         dl.FunctionIO('Json', value='running config')],
                        function_name='search')