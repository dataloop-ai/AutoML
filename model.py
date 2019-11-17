import json

class Model():
    def __init__(self, model):
        assert (model in self.list_available_models()), "we only have {} models".format(self.list_available_models())

    def build(self, json_path):
        with open(json_path) as f:
            self.build_dic = json.load(f)
        self.validate()
        weights = self.build_dic['weights']

    @staticmethod
    def list_available_models():
        return ['retinanet', 'yolo', 'mobilenet', 'example_model']

    def validate(self):
        if 'weights' not in self.build_dic:
            print('weights not in json, so weight initialized to small random')
        if 'hp_values' not in self.build_dic:
            raise Exception('hp_values must be defined in json')