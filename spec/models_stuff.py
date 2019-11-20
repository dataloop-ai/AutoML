from .spec_base import Spec


class ModelsSpec(Spec):

    def __init__(self, spec_data=None):
        if not spec_data:
            pass
        super().__init__(spec_data)
        pass
    def validate(self):
        # if 'model_space' not in self.spec_data:
        #     raise Exception("Model spec must have a model_space field")
        #
        # if 'task' not in self.spec_data:
        #     raise Exception("Recipe must have a task field")
        pass

    @property
    def models_space(self):
        new_dic = {}
        for model_name, model_dic in self.spec_data.items():
            new_dic[model_name] = []
            for rating in ['accuracy_rating', 'speed_rating', 'memory_rating']:
                new_dic[model_name].append(model_dic['model_space'][rating])

        return new_dic



    # @property
    # def task(self):
    #     return self.spec_data['task']