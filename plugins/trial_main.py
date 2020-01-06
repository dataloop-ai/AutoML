import logging
import dtlpy as dl
import numpy as np
from PIL import Image
import uuid
import os
from importlib import import_module
from .plugin_utils import download_data
logger = logging.getLogger(__name__)


class PluginRunner(dl.BasePluginRunner):
    """
    Plugin runner class

    """

    def __init__(self):
        pass

    def run(self, dataset_obj, model_specs, hp_values, configs=None, progress=None):

        download_data(dataset_obj)

        # get project
        project = dataset_obj.project
        assert isinstance(project, dl.entities.Project)

        # start tune
        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')
        final = 0
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