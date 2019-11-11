from .spec_base import Spec


class OptModel(Spec):

    @property
    def model_space(self):
        for dic in self.spec_data.values():
            if 'model_space' in dic:
                return dic['model_space']

    @property
    def task(self):
        for dic in self.spec_data.values():
            if 'task' in dic:
                return dic['task']
