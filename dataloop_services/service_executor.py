import logging
import os
import torch
import json
import dtlpy as dl
from importlib import import_module
from dataloop_services.plugin_utils import maybe_download_data
logger = logging.getLogger(__name__)


class ServiceRunner(dl.BaseServiceRunner):
    """
    Plugin runner class

    """

    def __init__(self, package_name, service_name):
        self.package_name = package_name
        self.service_name = service_name
        self.path_to_best_checkpoint = 'checkpoint.pt'
        self.path_to_metrics = 'metrics.json'
        self.path_to_tensorboard_dir = 'runs'
        logger.info(self.package_name + ' initialized')

    def run(self, dataset, model_specs, hp_values, configs=None, progress=None):

        maybe_download_data(dataset)

        # get project
        # project = dataset.project
        assert isinstance(dataset, dl.entities.Dataset)
        project = dl.projects.get(project_id=dataset.projects[0])

        # start tune
        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')

        final = 1 if self.service_name == 'trainer' else 0
        devices = {'gpu_index': 0}

        adapter = cls(devices, model_specs, hp_values, final)
        if hasattr(adapter, 'reformat'):
            adapter.reformat()
        if hasattr(adapter, 'data_loader'):
            adapter.data_loader()
        if hasattr(adapter, 'preprocess'):
            adapter.preprocess()
        if hasattr(adapter, 'build'):
            adapter.build()
        logger.info('commencing training . . . ')
        adapter.train()
        logger.info('training finished')
        save_info = {
            'package_name': self.package_name,
            'execution_id': progress.execution.id
        }
        if final:
            checkpoint = adapter.get_checkpoint()
            self._save_checkpoint(checkpoint)
            # upload checkpoint as artifact
            logger.info('uploading checkpoint to dataloop')
            project.artifacts.upload(filepath=self.path_to_best_checkpoint,
                                     package_name=save_info['package_name'],
                                     execution_id=save_info['execution_id'])

            project.artifacts.upload(filepath=self.path_to_tensorboard_dir,
                                     package_name=save_info['package_name'],
                                     execution_id=save_info['execution_id'])
            logger.info('finished uploading checkpoint and logs')
        else:
            metrics = adapter.get_metrics()
            if type(metrics) is not dict:
                raise Exception('adapter, get_metrics method must return dict object')
            if type(metrics['val_accuracy']) is not float:
                raise Exception(
                    'adapter, get_metrics method must return dict with only python floats. '
                    'Not numpy floats or any other objects like that')

            self._save_metrics(metrics)
            # upload metrics as artifact
            logger.info('uploading metrics to dataloop')
            project.artifacts.upload(filepath=self.path_to_metrics,
                                     package_name=save_info['package_name'],
                                     execution_id=save_info['execution_id'])

            project.artifacts.upload(filepath=self.path_to_tensorboard_dir,
                                     package_name=save_info['package_name'],
                                     execution_id=save_info['execution_id'])
            logger.info('finished uploading metrics and logs')

        logger.info('FINISHED SESSION')

    def _save_metrics(self, metrics):
        # save trial
        if os.path.exists(self.path_to_metrics):
            logger.info('overwriting checkpoint.pt . . .')
            try:
                os.remove(self.path_to_metrics)
            except IsADirectoryError:
                os.rmdir(self.path_to_metrics)
        with open(self.path_to_metrics, 'w') as fp:
            json.dump(metrics, fp)

    def _save_checkpoint(self, checkpoint):
        # save checkpoint
        if os.path.exists(self.path_to_best_checkpoint):
            logger.info('overwriting checkpoint.pt . . .')
            try:
                os.remove(self.path_to_best_checkpoint)
            except IsADirectoryError:
                os.rmdir(self.path_to_best_checkpoint)
        torch.save(checkpoint, self.path_to_best_checkpoint)