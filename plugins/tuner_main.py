import logging
import dtlpy as dl
import numpy as np
from PIL import Image
import uuid
import os

logger = logging.getLogger(__name__)


class PluginRunner(dl.BasePluginRunner):
    """
    Plugin runner class

    """

    def __init__(self):
        pass

    def run(self, dataset, configs=None, progress=None):
        project = dataset.project
        assert isinstance(project, dl.entities.Project)
        pipeline_id = str(uuid.uuid1())
        local_path = os.path.join(os.getcwd(), pipeline_id)
        #######################
        # download for resume #
        #######################
        project.artifacts.download(plugin_name='tuner',
                                   session_id=pipeline_id,
                                   local_path=local_path)

        # start tune

        # deploy trainers
        trainer_deployment = project.deployments.get(deployment_name='trainer')
        for i in range(5):
            trainer_deployment.sessions.create()

        #####################
        # upload for resume #
        #####################
        project.artifacts.upload(plugin_name='tuner',
                                 session_id=pipeline_id,
                                 local_path=local_path)
