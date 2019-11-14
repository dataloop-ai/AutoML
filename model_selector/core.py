import numpy as np


class ModelSelector:

    def __init__(self, optimal_model):
        self.optimal_model = optimal_model
        self.model_space = np.array(optimal_model.model_space)
        self.task = optimal_model.task
        self.model_space_dic = {
                                    'retinanet': np.array((8, 2, 4)),
                                    'yolo': np.array((1, 10, 4)),
                                    'mobilenet': np.array((1, 4, 10))
        }
        self.distance_from_dic = {}

    def find_model_and_hp_search_space(self):

        for model, space in self.model_space_dic.items():
            self.distance_from_dic[model] = np.linalg.norm(self.model_space - space)
        closest_model = min(self.distance_from_dic.keys(), key=(lambda x: self.distance_from_dic[x]))

        hp_search_space = [{"name": "input_size", "default": None, "values": [14, 28, 56],
                            "step": 1, "sampling": None},

                           {"name": "learning_rate", "default": 0.01,
                            "values": [0.01, 0.001, 0.0001], "ordered": None}]

        self.optimal_model.add_attr(closest_model, 'model')
        self.optimal_model.add_attr(hp_search_space, 'hp_space')
