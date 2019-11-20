import numpy as np
from spec import ModelsSpec
import os


class ModelSelector:

    def __init__(self, optimal_model):
        self.optimal_model = optimal_model
        this_path = os.getcwd()
        models_spec_path = os.path.join(this_path, 'models.json')
        self.models = ModelsSpec(models_spec_path)

        self.model_space = np.array(optimal_model.model_priority_space)
        self.task = optimal_model.task
        self.model_space_dic = self.models.models_space

    def find_model_and_hp_search_space(self):
        distance_from_dic = {}
        for model, space in self.model_space_dic.items():
            distance_from_dic[model] = np.linalg.norm(self.model_space - space)
        closest_model = min(distance_from_dic.keys(), key=(lambda x: distance_from_dic[x]))

        self.optimal_model.add_attr(closest_model, 'model')
        self.optimal_model.add_attr(self.models.spec_data[closest_model]['hp_search_space'], 'hp_space')
        self.optimal_model.add_attr(self.models.spec_data[closest_model]['training_configs'], 'training_configs')
