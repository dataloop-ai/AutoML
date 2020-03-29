import dtlpy as dl
import logging
import json
from logging_utils import logginger
logger = logging.getLogger(name=__name__)
from importlib import import_module


class LocalPredConnector():

    def __init__(self):
        self.logger = logginger(__name__)

    def run(self, checkpoint_path, model_specs):
        cls = getattr(import_module('.adapter', 'zoo.' + model_specs['name']), 'AdapterModel')

        home_path = model_specs['data']['home_path']

        inputs_dict = {'checkpoint_path': checkpoint_path, 'home_path': home_path}
        with open('predict_configs.json', 'w') as fp:
            json.dump(inputs_dict, fp)

        adapter = cls()
        adapter.predict()
