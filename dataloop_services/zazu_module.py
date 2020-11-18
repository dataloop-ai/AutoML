import logging
import os
import sys
import glob
import dtlpy as dl
from spec import ConfigSpec, OptModel
from zazu import ZaZu
from logging_utils import logginger, init_logging

logger = init_logging(__name__)

class ServiceRunner(dl.BaseServiceRunner):
    """
    Plugin runner class

    """

    def __init__(self, package_name):
        logging.getLogger('dtlpy').setLevel(logging.WARN)
        self.package_name = package_name
        self.this_path = os.getcwd()
        logger.info(self.package_name + ' initialized')

    def search(self, configs, progress=None):

        configs = ConfigSpec(configs)
        opt_model = OptModel()
        opt_model.add_child_spec(configs, 'configs')
        zazu = ZaZu(opt_model, remote=True)
        zazu.find_best_model()
        zazu.search()
        checkpoint_paths_list = glob.glob('*checkpoint*.pt')
        save_info = {
            'package_name': self.package_name,
            'execution_id': progress.execution.id
        }

        project_name = opt_model.dataloop['project']
        project = dl.projects.get(project_name=project_name)

        # model_name = opt_model.name
        # model_obj = dl.models.get(model_name=model_name)
        logger.info('uploading checkpoints.....')
        for checkpoint_path in checkpoint_paths_list:
            # model_obj.checkpoints.upload(checkpoint_name=checkpoint_path.split('.')[0], local_path=checkpoint_path)
            project.artifacts.upload(filepath=checkpoint_path,
                                     package_name=save_info['package_name'],
                                     execution_id=save_info['execution_id'])

        logger.info('finished uploading checkpoints')


    def predict(self, configs, progress=None):

        configs = ConfigSpec(configs)
        opt_model = OptModel()
        opt_model.add_child_spec(configs, 'configs')
        zazu = ZaZu(opt_model, remote=True)
        zazu.run_inference()
