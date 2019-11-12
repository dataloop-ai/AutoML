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
    def items_local_path(self):
        for dic in self.spec_data.values():
            if 'items_local_path' in dic:
                return dic['items_local_path']

        return None

    @property
    def labels_local_path(self):
        for dic in self.spec_data.values():
            if 'labels_local_path' in dic:
                return dic['labels_local_path']

        return None

    @property
    def remote_dataset_id(self):
        for dic in self.spec_data.values():
            if 'remote_dataset_id' in dic:
                return dic['remote_dataset_id']

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
