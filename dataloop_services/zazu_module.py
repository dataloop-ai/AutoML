import logging
import os
import dtlpy as dl
from spec import ConfigSpec, OptModel
from zazu import ZaZu
logger = logging.getLogger(__name__)


class ServiceRunner(dl.BaseServiceRunner):
    """
    Plugin runner class

    """

    def __init__(self, package_name):
        self.package_name = package_name
        self.this_path = os.getcwd()
        logger.info(self.package_name + ' initialized')

    def search(self, configs, progress=None):

        configs = ConfigSpec(configs)
        opt_model = OptModel()
        opt_model.add_child_spec(configs, 'configs')
        zazu = ZaZu(opt_model, remote=True)
        zazu.find_best_model()
        zazu.hp_search()

        save_info = {
            'package_name': self.package_name,
            'execution_id': progress.execution.id
        }

        project_name = opt_model.dataloop['project']
        project = dl.projects.get(project_name=project_name)

        paths = [zazu.path_to_most_suitable_model, zazu.path_to_best_trial, zazu.path_to_best_checkpoint]
        for path in paths:
            project.artifacts.upload(filepath=path,
                                     package_name=save_info['package_name'],
                                     execution_id=save_info['execution_id'])

    def train(self, configs, progress=None):

        configs = ConfigSpec(configs)
        opt_model = OptModel()
        opt_model.add_child_spec(configs, 'configs')
        zazu = ZaZu(opt_model, remote=True)
        zazu.train_new_model()

        save_info = {
            'package_name': self.package_name,
            'execution_id': progress.execution.id
        }

        project_name = opt_model.dataloop['project']
        project = dl.projects.get(project_name=project_name)

        paths = [zazu.path_to_most_suitable_model, zazu.path_to_best_trial, zazu.path_to_best_checkpoint]
        for path in paths:
            project.artifacts.upload(filepath=path,
                                     package_name=save_info['package_name'],
                                     execution_id=save_info['execution_id'])

    def predict(self, configs, progress=None):

        configs = ConfigSpec(configs)
        opt_model = OptModel()
        opt_model.add_child_spec(configs, 'configs')
        zazu = ZaZu(opt_model, remote=True)
        zazu.run_inference()