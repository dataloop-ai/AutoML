from .spec_base import Spec


class ModelSpaceSpec(Spec):

    def __init__(self, spec_data=None):
        if not spec_data:
            spec_data = {"model_space": (10, 0, 0)}
        super().__init__(spec_data)

    def validate(self):
        if 'model_space' not in self.spec_data:
            raise Exception("Model spec must have a model_space field")

    @property
    def model_space(self):
        return self.spec_data['model_space']