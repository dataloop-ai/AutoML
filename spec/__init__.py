import os.path
import json
from enum import Enum


class SPEC_TYPE(Enum):
    MODEL_SELECTION = "model_selection"


class MODULE_TYPE(Enum):
    RECIPE = "recipe"


class SpecModule:
    def __init__(self):
        self._moduleData = {}

    @property
    def name(self):
        if not self._moduleData['name']:
            return ''
        return self._moduleData['name']

    @property
    def type(self):
        if not self._moduleData['type']:
            return ''
        return self._moduleData['type']


class Spec:
    def __init__(self,spec_data):
        self.spec_data_={}
        self.load(spec_data)

    def load(self, dict_or_spec_file_path):
        if os.path.isfile(dict_or_spec_file_path):
            with open(dict_or_spec_file_path) as f:
                spec_data = json.load(f)
        else:
            spec_data = dict_or_spec_file_path
        self.spec_data_ = spec_data
        self.validate()

    def save(self, file_path_name):
        # covert object state to dict
        state = self.get_state()
        with open(file_path_name, "w") as f:
            json.dump(state, f)

    @property
    def name(self):
        if not 'name' in self.spec_data_:
            return ''
        return self.spec_data_['name']

    @property
    def uuid(self):
        if not 'uuid' in self.spec_data_:
            return 'NA'
        return self.spec_data_['uuid']

    @property
    def type(self):
        if not 'type' in self.spec_data_:
            return 'NA'
        return self.spec_data_['type']

    @property
    def specs(self):
        if not 'specs' in self.spec_data_:
            return []
        return self.spec_data_['specs']

    def validate(self):
        pass

class RecipeSpec(Spec):

    def validate(self):
        if not 'task' in self.spec_data_:
            raise Exception("Recipe must have a task field")

    @property
    def task(self):
        return self.spec_data_['task']

class ModelSelectionSpec(Spec):
    def validate(self):
        # code to validate here ...
        pass


class Trial(Spec):
    def __init__(self, trial_id, hp_values, status):
        self.trial_id = trial_id
        self.hp_values = hp_values
        self.status = status
        self.metrics = {}

    def load_state(self, state):
        self.trial_id = state['trial_id']
        self.hp_values = state['hp_values'] # dict at first
        self.metrics = state['metrics']
        self.status = state['status']

    def get_state(self):
        state_dict = {
            'trial_id': self.trial_id,
            'hp_values': self.hp_values,
            'metrics': self.metrics,
            'status': self.status
        }
        return state_dict

class Oracle:
    pass


class OngoingTrial:
    pass


class Metric:
    pass


class HpValues:
    pass


class SearchSpace:
    pass
