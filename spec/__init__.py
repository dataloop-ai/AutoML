import os.path
import json
from enum import Enum
class SPEC_TYPE(Enum):
    MODEL_SELECTION="model_selection"

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
        self._specData = {}
        self.modules = []

    def load_state(self, state):
        self._specData = state
        for i in state['modules']:
            self.modules.append()

    def reload(self, file_path_name):
        if not self.validate():
            raise Exception("Faild validation")
        if os.path.isfile(file_path_name):
            with open(file_path_name) as f:
                state = json.load(f)
        self.loadState(state)


    def save(self, file_path_name):
        # covert object state to dict
        state = self.get_state()
        with open(file_path_name, "w") as f:
            json.dump(state, f)

    def validate(self):
        pass

    @property
    def modules_json(self):
        if not self._specData['modules']:
            return []
        return self._specData['modules']

    @property
    def modulesNum(self):
        return len(self.modules)

    def getModuleByType(self):
        pass


class ModelSelectionSpec(Spec):
    def validate(self):
        #code to validate here ...
        pass


class Trial:
    pass


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
