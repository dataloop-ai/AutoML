import numpy as np
from spec import ModelsSpec
import os


def find_model(optimal_model, models):
    this_path = os.getcwd()
    models_spec_path = os.path.join(this_path, 'models.json')
    models = ModelsSpec(models_spec_path)

    model_space = np.array(optimal_model.model_priority_space)
    task = optimal_model.task
    model_space_dic = models.models_space

    distance_from_dic = {}
    for model, space in model_space_dic.items():
        distance_from_dic[model] = np.linalg.norm(model_space - space)
    closest_model = min(distance_from_dic.keys(), key=(lambda x: distance_from_dic[x]))
    return closest_model

