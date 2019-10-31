
class Spinner:
    def __init__(self, task, priority):
        self.task = task
        self.priority = priority

    def find_closest_model_and_hp(self):
        model = 'retinanet'

        configs = {"max_trials": 5, "epochs": 10, "max_instances_at_once": 2, "data_local_path": "/Users/noam/searchJob/mini_coco",
                   "data_remote_id": 32409823}

        search_space = [{"name": "input_size", "default": None, "values": [14,28,56],
                                              "step": 1, "sampling": None},

                                  {"name": "learning_rate", "default": 0.01,
                                              "values": [0.01, 0.001, 0.0001], "ordered": None}]
        return search_space, model, configs



