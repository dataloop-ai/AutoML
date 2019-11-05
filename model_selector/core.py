
class ModelSelector:
    def __init__(self, optimal_model):
        self.optimal_model = optimal_model
        self.model_space = optimal_model.model_space
        self.task = optimal_model.task

    def find_model_and_hp_search_space(self):
        model = 'retinanet'

        configs = {"max_trials": 5, "epochs": 10, "max_instances_at_once": 2, "data_local_path": "/Users/noam/searchJob/mini_coco",
                   "data_remote_id": 32409823}

        hp_search_space = [{"name": "input_size", "default": None, "values": [14,28,56],
                                              "step": 1, "sampling": None},

                                  {"name": "learning_rate", "default": 0.01,
                                              "values": [0.01, 0.001, 0.0001], "ordered": None}]
        self.optimal_model.add_attr(model, 'model')
        self.optimal_model.add_attr(configs, 'configs')
        self.optimal_model.add_attr(hp_search_space, 'hp_space')




