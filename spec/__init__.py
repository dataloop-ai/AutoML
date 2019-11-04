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
    def __init__(self):
        pass

    def load_state(self, state):
        for specific_state in state:
            pass

    def load(self, file_path_name):
        if os.path.isfile(file_path_name):
            with open(file_path_name) as f:
                state = json.load(f)
            self.load_state(state)

    def save(self, file_path_name):
        # covert object state to dict
        state = self.get_state()
        with open(file_path_name, "w") as f:
            json.dump(state, f)


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
