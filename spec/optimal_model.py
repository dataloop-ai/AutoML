from .spec_base import Spec
import os
import dtlpy as dl

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
            if self.dataloop:
                return {'home_path': os.path.join('..', 'data', self.dataloop['dataset']), 'annotation_type': 'coco', 'dataset_name': ''}
            elif 'data' in dic:
                return dic['data']
            else:
                return None

    @property
    def dataloop(self):
        for dic in self.spec_data.values():
            if 'dataloop' in dic:
                try:
                    dic['dataloop']['project']
                    dic['dataloop']['dataset']
                except:
                    project_id = dic['dataloop']['project_id']
                    dataset_id = dic['dataloop']['dataset_id']
                    project = dl.projects.get(project_id=project_id)
                    dataset = project.datasets.get(dataset_id=dataset_id)
                    project_name = project.name
                    dataset_name = dataset.name
                    dic['dataloop']['project'] = project_name
                    dic['dataloop']['dataset'] = dataset_name
                return dic['dataloop']



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
            'name': self.name,
            'training_configs': self.training_configs,
            'data': self.data
        }
