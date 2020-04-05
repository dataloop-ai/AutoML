import dtlpy as dl
import logging
import os
import json
import torch
from logging_utils import logginger
logger = logging.getLogger(name=__name__)
from importlib import import_module
from dataloop_services.plugin_utils import maybe_download_pred_data


class ServiceRunner(dl.BaseServiceRunner):
    """
    Plugin runner class

    """
    def __init__(self, package_name):
        self.package_name = package_name
        self.logger = logginger(__name__)

    def run(self, dataset, val_query, checkpoint_path, model_specs, configs=None, progress=None):
        self.logger.info('checkpoint path: ' + str(checkpoint_path))
        self.logger.info('Beginning to download checkpoint')
        dataset.items.get(filepath='/checkpoints').download(local_path=os.getcwd())
        self.logger.info('checkpoint downloaded, dir is here' + str(os.listdir('.')))
        self.logger.info('downloading data')
        maybe_download_pred_data(dataset, val_query)
        self.logger.info('data downloaded')
        assert isinstance(dataset, dl.entities.Dataset)
        project = dl.projects.get(project_id=dataset.projects[0])
        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')

        home_path = model_specs['data']['home_path']

        inputs_dict = {'checkpoint_path': checkpoint_path['checkpoint_path'], 'home_path': home_path}
        torch.save(inputs_dict, 'predict_checkpoint.pt')

        adapter = cls()
        output_path = adapter.predict(home_path=home_path, checkpoint_path=checkpoint_path['checkpoint_path'])
        save_info = {
            'package_name': self.package_name,
            'execution_id': progress.execution.id
        }
        project.artifacts.upload(filepath=output_path,
                                 package_name=save_info['package_name'],
                                 execution_id=save_info['execution_id'])