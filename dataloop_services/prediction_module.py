import dtlpy as dl
import os
import logging
from importlib import import_module
import torch
logger = logging.getLogger(__name__)


class ServiceRunner(dl.BaseServiceRunner):
    """
    Package runner class

    """

    def __init__(self, model_id, checkpoint_id):
        """
        Init package attributes here

        :param kwargs: config params
        :return:
        """
        # project = dl.projects.get('buffs_project')
        # model = project.models.get(model_id=model_id)
        model = dl.models.get(model_id=model_id)
        self.name = model.name
        self.adapter = model.build(local_path=os.path.join(os.getcwd(), self.name))
        self.load_new_inference_checkpoint(model_id=model_id,
                                           checkpoint_id=checkpoint_id)

    def load_new_inference_checkpoint(self, model_id, checkpoint_id, progress=None):
        self.adapter.load_from_inference_checkpoint(model_id=model_id,
                                                    checkpoint_id=checkpoint_id)
        logger.info('new checkpoint loaded into adapter object')

    def predict_single_item(self, item, progress=None):
        print('has cuda: ' + str(torch.cuda.is_available()))
        logger.info('has cuda: ' + str(torch.cuda.is_available()))
        dirname = self.adapter.predict_item(item, model_name=self.name)

        logger.info('uploaded prediction from ' + dirname)


if __name__ == "__main__":
    """
    Run this main to locally debug your package
    """
    dl.packages.test_local_package()
