import logging
import os
import torch
import json
import dtlpy as dl
from importlib import import_module
from dataloop_services.plugin_utils import maybe_download_data
from logging_utils import init_logging


class ServiceRunner(dl.BaseServiceRunner):
    """
    Plugin runner class

    """

    def __init__(self, package_name):
        logging.getLogger('dtlpy').setLevel(logging.WARN)
        self.package_name = package_name
        self.path_to_metrics = 'metrics.json'
        self.path_to_tensorboard_dir = 'runs'
        self.path_to_logs = 'logger.conf'
        self.logger = init_logging(__name__, filename=self.path_to_logs)
        self.logger.info(self.package_name + ' initialized')


    def run(self, dataset, train_query, val_query, model_specs, hp_values, configs=None, progress=None):
        maybe_download_data(dataset, train_query, val_query)

        # get project
        # project = dataset.project
        assert isinstance(dataset, dl.entities.Dataset)
        project = dl.projects.get(project_id=dataset.projects[0])

        # start tune
        cls = getattr(import_module('.adapter', 'ObjectDetNet.' + model_specs['name']), 'AdapterModel')

        inputs_dict = {'devices': {'gpu_index': 0}, 'model_specs': model_specs, 'hp_values': hp_values}
        torch.save(inputs_dict, 'checkpoint.pt')

        adapter = cls()
        adapter.load()
        if hasattr(adapter, 'reformat'):
            adapter.reformat()
        if hasattr(adapter, 'data_loader'):
            adapter.data_loader()
        if hasattr(adapter, 'preprocess'):
            adapter.preprocess()
        if hasattr(adapter, 'build'):
            adapter.build()
        self.logger.info('commencing training . . . ')
        adapter.train()
        self.logger.info('training finished')
        save_info = {
            'package_name': self.package_name,
            'execution_id': progress.execution.id
        }
        checkpoint_path = adapter.save()

        # upload metrics as artifact
        self.logger.info('uploading metrics to dataloop')
        project.artifacts.upload(filepath=self.path_to_logs,
                                 package_name=save_info['package_name'],
                                 execution_id=save_info['execution_id'])

        # this is the same as uplading metrics because the map is saved under checkpoint['metrics']['val_accuracy']
        project.artifacts.upload(filepath=checkpoint_path,
                                 package_name=save_info['package_name'],
                                 execution_id=save_info['execution_id'])

        project.artifacts.upload(filepath=self.path_to_tensorboard_dir,
                                 package_name=save_info['package_name'],
                                 execution_id=save_info['execution_id'])

        adapter.delete_stuff()
        self.logger.info('finished uploading checkpoint and logs')

        self.logger.info('FINISHED SESSION')
