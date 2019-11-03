import os.path
import json


class Spec:
    def __init__(self):
        self._specData = {}

    def reload(self, file_path_name):
        if os.path.isfile(file_path_name):
            with open(file_path_name) as f:
                state = json.load(f)
        self.set_state(state)

    def save(self, file_path_name):
        # covert object state to dict
        state = self.get_state()
        with open(file_path_name, "w") as f:
            json.dump(state, f)


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
