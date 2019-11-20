from .spec_base import Spec


class OptModel(Spec):

    @property
    def model_space(self):
        for dic in self.spec_data.values():
            if 'model_space' in dic:
                return dic['model_space']

        return None

    @property
    def task(self):
        for dic in self.spec_data.values():
            if 'task' in dic:
                return dic['task']

        return None

    @property
    def data(self):
        for dic in self.spec_data.values():
            if 'data' in dic:
                return dic['data']

        return None

    @property
    def max_trials(self):
        for dic in self.spec_data.values():
            if 'max_trials' in dic:
                return dic['max_trials']

        return None

    @property
    def max_instances_at_once(self):
        for dic in self.spec_data.values():
            if 'max_instances_at_once' in dic:
                return dic['max_instances_at_once']

        return None

    @property
    def model_priority_space(self):
        for dic in self.spec_data.values():
            if 'model_priority_space' in dic:
                return dic['model_priority_space']

        return None

    def unwrap(self):
        return {
            'model': self.model,
            'training_configs': self.training_configs,
            'data': self.data
        }
