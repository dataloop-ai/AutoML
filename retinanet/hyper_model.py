import os
from dl_to_csv import create_annotations_txt
from .train import train


class HyperModel:

    def __init__(self):
        self.path = os.getcwd()
        self.output_path = os.path.join(self.path, 'output')

    def data_loader(self, configs):
        labels_list = ['kite', 'dog', 'person', 'bird']
        classes_filepath = os.path.join(self.output_path, 'classes.txt')
        annotations_train_filepath = os.path.join(self.output_path, 'annotations_train.txt')
        annotations_val_filepath = os.path.join(self.output_path, 'annotations_val.txt')
        create_annotations_txt(annotations_path=configs["labels_local_path"],
                                    images_path=configs["items_local_path"],
                                    train_split=0.9,
                                    train_filepath=annotations_train_filepath,
                                    val_filepath=annotations_val_filepath,
                                    classes_filepath=classes_filepath,
                                    labels_list=labels_list)
        train('csv', csv_train=annotations_train_filepath, csv_val=annotations_val_filepath, csv_classes=classes_filepath)

    def add_preprocess(self, hp_values):
        pass

    def build(self, hp_values):
        pass

    def train(self):
        pass

    def infer(self):
        pass

    def eval(self):
        pass
