from .spec_base import Spec


class DataSpec(Spec):

    def __init__(self, spec_data=None):
        super().__init__(spec_data)

    def validate(self):
        pass
