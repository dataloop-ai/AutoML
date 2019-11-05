import os
import json


class Spec:

    def __init__(self, spec_data):
        if spec_data:
            self.load(spec_data)

    def load(self, dict_or_spec_file_path):
        # if str and path exists
        if isinstance(dict_or_spec_file_path, str) and os.path.isfile(dict_or_spec_file_path):
            with open(dict_or_spec_file_path) as f:
                spec_data = json.load(f)
        else:
            # assume its a dict
            spec_data = dict_or_spec_file_path
        # turn into class attributes
        for spec_name, spec_body in spec_data.items():
            setattr(self, spec_name, spec_body)

        self.validate()

    def save(self, file_path_name):
        # covert object state to dict
        state = self.get_state()
        with open(file_path_name, "w") as f:
            json.dump(state, f)

    def validate(self):
        pass
