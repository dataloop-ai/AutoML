import logging
import dtlpy as dl
import numpy as np
from PIL import Image
import os

logger = logging.getLogger(__name__)


class PluginRunner(dl.BasePluginRunner):
    """
    Plugin runner class

    """

    def __init__(self):
        logger.info('UUUUUUUUUUUUUPPPPPPPPPPPPPPPP')

    def run(self, dataset, config=None, progress=None):
        logger.info('RRRRRRRRUUUUUUUUUUNNNNNNNNNN')
