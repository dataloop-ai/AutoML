from .spec_base import Spec


class DataSpec(Spec):

    def __init__(self, spec_data=None):
        if not spec_data:
            spec_data = {'data_type': "unknown"}
        super().__init__(spec_data)
        self.items = []
        self.labels = []

    def validate(self):
        if 'data_type' not in self.spec_data:
            raise Exception("Missing data type")

    def fill(self, items, labels):
        self.items = items
        self.labels = labels

