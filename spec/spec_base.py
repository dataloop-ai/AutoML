import os
import json


class Spec:

    def __init__(self, spec_data=None):
        if spec_data:
            self.load(spec_data)
        else:
            self.spec_data = {}

    def load(self, dict_or_spec_file_path):
        # if str and path exists
        if isinstance(dict_or_spec_file_path, str) and os.path.isfile(dict_or_spec_file_path):
            with open(dict_or_spec_file_path) as f:
                spec_data = json.load(f)
        else:
            # assume its a dict
            spec_data = dict_or_spec_file_path
        # turn into class attributes
        self.spec_data = spec_data
        self.validate()

    def save(self, file_path_name):
        # covert object state to dict
        state = self.get_state()
        with open(file_path_name, "w") as f:
            json.dump(state, f)

    def add_child_spec(self, obj, name):
        self.spec_data[name] = obj.spec_data

    def add_attr(self, value, name):
        setattr(self, name, value)

    def add_attr_from_obj(self, obj, name):
        setattr(self, name, getattr(obj, name))

    def validate(self):
        pass
