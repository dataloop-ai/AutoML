import logging
import os
import torch
import dtlpy as dl
from importlib import import_module
from plugin_utils import maybe_download_data
logger = logging.getLogger(__name__)


class PluginRunner(dl.BasePluginRunner):
    """
    Plugin runner class

    """

    def __init__(self, plugin_name):
        self.plugin_name = plugin_name
        self.path_to_best_checkpoint = 'checkpoint.pt'

    def run(self, dataset, model_specs, hp_values, configs=None, progress=None):

        maybe_download_data(dataset)

        # get project
        # project = dataset.project
        assert isinstance(dataset, dl.entities.Dataset)
        project = dl.projects.get(project_id=dataset.projects[0])

        # start tune
        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')

        final = 1 if self.plugin_name == 'trainer' else 0
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
        adapter.train()

        if final:
            checkpoint = adapter.get_checkpoint()
            # save checkpoint and upload as artifact
            if os.path.exists(self.path_to_best_checkpoint):
                print('overwriting checkpoint.pt . . .')
                os.remove(self.path_to_best_checkpoint)
            torch.save(checkpoint, self.path_to_best_checkpoint)
            checkpoint_save_info = {
                'plugin_name': self.plugin_name,
                'session_id': progress.session.id
                                }
            project.artifacts.upload(filepath=self.path_to_best_checkpoint,
                                     plugin_name=checkpoint_save_info['plugin_name'],
                                     session_id=checkpoint_save_info['session_id'])
            return checkpoint_save_info
        else:
            metrics = adapter.get_metrics()
            if type(metrics) is not dict:
                raise Exception('adapter, get_metrics method must return dict object')
            if type(metrics['val_accuracy']) is not float:
                raise Exception(
                    'adapter, get_metrics method must return dict with only python floats. '
                    'Not numpy floats or any other objects like that')
            return metrics

        # pipeline_id = str(uuid.uuid1())
        # local_path = os.path.join(os.getcwd(), pipeline_id)
        #
        # #####################
        # # upload for resume #
        # #####################
        # project.artifacts.upload(plugin_name='tuner',
        #                          session_id=pipeline_id,
        #                          local_path=local_path)
        #
        # #######################
        # # download for resume #
        # #######################
        # project.artifacts.download(plugin_name='tuner',
        #                            session_id=pipeline_id,
        #                            local_path=local_path)