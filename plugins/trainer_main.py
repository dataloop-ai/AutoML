import logging
import dtlpy as dl
import numpy as np
from PIL import Image
import os
from .plugin_utils import download_data
from importlib import import_module
logger = logging.getLogger(__name__)


class PluginRunner(dl.BasePluginRunner):
    """
    Plugin runner class

    """

    def __init__(self):
        pass

    def run(self, dataset_obj, model_specs, hp_values, config=None, progress=None):

        download_data(dataset_obj)

        # get project
        project = dataset_obj.project
        assert isinstance(project, dl.entities.Project)

        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')
        final = 1
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

        return adapter.get_checkpoint()
