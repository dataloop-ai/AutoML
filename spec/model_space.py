from .spec_base import Spec


class ModelSpaceSpec(Spec):

    def __init__(self, spec_data=None):
        if not spec_data:
            # default model space
            spec_data = {"model_space": (10, 0, 0), "task": "detection"}
            print("model_space set to default high accuracy high latency")
        super().__init__(spec_data)

    def validate(self):
        if 'model_space' not in self.spec_data:
            raise Exception("Model spec must have a model_space field")

        if 'task' not in self.spec_data:
            raise Exception("Recipe must have a task field")

    @property
    def model_space(self):
        return self.spec_data['model_space']

    @property
    def task(self):
        return self.spec_data['task']