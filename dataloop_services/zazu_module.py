import logging
import os
import torch
import json
import dtlpy as dl
from importlib import import_module
from plugin_utils import maybe_download_data
from spec import ConfigSpec, OptModel
from spec import ModelsSpec
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

    def search(self, progress=None):
        configs_path = os.path.join(self.this_path, 'configs.json')
        configs = ConfigSpec(configs_path)
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


    def train(self, progress=None):
        configs_path = os.path.join(self.this_path, 'configs.json')
        configs = ConfigSpec(configs_path)
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

    def predict(self, progress=None):
        configs_path = os.path.join(self.this_path, 'configs.json')
        configs = ConfigSpec(configs_path)
        opt_model = OptModel()
        opt_model.add_child_spec(configs, 'configs')
        zazu = ZaZu(opt_model, remote=True)
        zazu.run_inference()


if __name__ == "__main__":

    logger.info('dtlpy version:', dl.__version__)
    try:
        dl.setenv('dev')
    except:
        dl.login()
        dl.setenv('dev')
    project = dl.projects.get('buffs_project')
    package_name = 'zazuml'

    init_specs_input = dl.FunctionIO(type='Json', name='package_name')
    input_to_init = {
        'package_name': package_name
    }

    inputs = []
    train_function = dl.PackageFunction(name='train', inputs=inputs, outputs=[], description='')
    search_function = dl.PackageFunction(name='search', inputs=inputs, outputs=[], description='')
    module = dl.PackageModule(entry_point='dataloop_services/zazu_module.py', name='zazu_module',
                              functions=[train_function, search_function],
                              init_inputs=init_specs_input)

    package = project.packages.push(
        package_name=package_name,
        src_path=os.path.dirname(os.getcwd()),
        modules=[module])

    logger.info('deploying package . . .')
    service = package.services.deploy(service_name=package.name,
                                           module_name='zazu_module',
                                           agent_versions={
                                               'dtlpy': '1.9.7',
                                               'runner': '1.9.7.latest',
                                               'proxy': '1.9.7.latest',
                                               'verify': True
                                           },
                                           package=package,
                                           runtime={'gpu': False,
                                                    'numReplicas': 1,
                                                    'concurrency': 2,
                                                    'runnerImage': 'buffalonoam/zazu-image:0.3'
                                                    },
                                           init_input=input_to_init)

    zazu_service = dl.services.get('zazuml')
    zazu_service.execute(function_name='search')