from .spec_base import Spec
import os
import dtlpy as dl
import json
class OptModel(Spec):

    def __init__(self, models_config_location):
        self.models_config_location = models_config_location
        super().__init__()

    @property
    def hp_space(self):
        with open(self.models_config_location) as f:
            models_data = json.load(f)
        return models_data[self.name]['hp_search_space']

    @property
    def training_configs(self):
        with open(self.models_config_location) as f:
            models_data = json.load(f)
        return models_data[self.name]['training_configs']
