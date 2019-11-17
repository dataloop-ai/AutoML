from .train import train

class HyperModel:
    def __init__(self):
        pass

    def data_loader(self, configs):
        pass

    def add_preprocess(self, hp_values):
        pass

    def build(self, hp_values):
        pass

    def train(self):
        train(args=['--coco_path', '/home/noam/data/coco'])

    def infer(self):
        pass

    def eval(self):
        pass
