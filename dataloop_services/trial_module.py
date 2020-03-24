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

    def __init__(self, package_name, service_name):
        logging.getLogger('dtlpy').setLevel(logging.WARN)
        self.package_name = package_name
        self.service_name = service_name
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
        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')

        devices = {'gpu_index': 0}

        adapter = cls()
        adapter.trial_init(devices, model_specs, hp_values)
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

        metrics_and_checkpoint_dict = adapter.get_best_metrics_and_checkpoint()
        if type(metrics_and_checkpoint_dict) is not dict:
            raise Exception('adapter, get_best_metrics method must return dict object')
        if type(metrics_and_checkpoint_dict['metrics']['val_accuracy']) is not float:
            raise Exception(
                'adapter, get_best_metrics method must return dict with only python floats. '
                'Not numpy floats or any other objects like that')

        # upload metrics as artifact
        self.logger.info('uploading metrics to dataloop')
        project.artifacts.upload(filepath=self.path_to_logs,
                                 package_name=save_info['package_name'],
                                 execution_id=save_info['execution_id'])

        # this is the same as uplading metrics because the map is saved under checkpoint['metrics']['val_accuracy']
        project.artifacts.upload(filepath=adapter.path_to_best_checkpoint,
                                 package_name=save_info['package_name'],
                                 execution_id=save_info['execution_id'])

        project.artifacts.upload(filepath=self.path_to_tensorboard_dir,
                                 package_name=save_info['package_name'],
                                 execution_id=save_info['execution_id'])
        self.logger.info('finished uploading metrics and logs')

        self.logger.info('FINISHED SESSION')

    def _save_metrics(self, metrics):
        # save trial
        if os.path.exists(self.path_to_metrics):
            self.logger.info('overwriting checkpoint.pt . . .')
            try:
                os.remove(self.path_to_metrics)
            except IsADirectoryError:
                os.rmdir(self.path_to_metrics)
        with open(self.path_to_metrics, 'w') as fp:
            json.dump(metrics, fp)

    def _save_checkpoint(self, checkpoint):
        # save checkpoint
        if os.path.exists(self.path_to_best_checkpoint):
            self.logger.info('overwriting checkpoint.pt . . .')
            try:
                os.remove(self.path_to_best_checkpoint)
            except IsADirectoryError:
                os.rmdir(self.path_to_best_checkpoint)
        torch.save(checkpoint, self.path_to_best_checkpoint)