import os
from dl_to_csv import create_annotations_txt
from .train import train


class HyperModel:

    def __init__(self, model, hp_values):
        self.model = model
        self.hp_values = hp_values
        self.path = os.getcwd()
        self.output_path = os.path.join(self.path, 'output')

        self.classes_filepath = os.path.join(self.output_path, 'classes.txt')
        self.annotations_train_filepath = os.path.join(self.output_path, 'annotations_train.txt')
        self.annotations_val_filepath = os.path.join(self.output_path, 'annotations_val.txt')

    def data_loader(self):
        labels_list = self.model['training_configs']['labels_list']
        local_labels_path = os.path.join(self.path, self.model['data']['labels_relative_path'])
        local_items_path = os.path.join(self.path, self.model['data']['items_relative_path'])

        create_annotations_txt(annotations_path=local_labels_path,
                               images_path=local_items_path,
                               train_split=0.9,
                               train_filepath=self.annotations_train_filepath,
                               val_filepath=self.annotations_val_filepath,
                               classes_filepath=self.classes_filepath,
                               labels_list=labels_list)

    def add_preprocess(self, hp_values):
        pass

    def build(self, hp_values):
        pass

    def train(self):
        train('csv', csv_train=self.annotations_train_filepath, csv_val=self.annotations_val_filepath,
              csv_classes=self.classes_filepath, epochs=self.model['training_configs']['epochs'], resize_min_side=self.hp_values['input_size'])

    def infer(self):
        pass

    def eval(self):
        pass
