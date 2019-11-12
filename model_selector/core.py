class ModelSelector:
    def __init__(self, optimal_model):
        self.optimal_model = optimal_model
        self.model_space = optimal_model.model_space
        self.task = optimal_model.task

    def find_model_and_hp_search_space(self):
        model = 'retinanet'

        hp_search_space = [{"name": "input_size", "default": None, "values": [14, 28, 56],
                            "step": 1, "sampling": None},

                           {"name": "learning_rate", "default": 0.01,
                            "values": [0.01, 0.001, 0.0001], "ordered": None}]

        self.optimal_model.add_attr(model, 'model')
        self.optimal_model.add_attr(hp_search_space, 'hp_space')
