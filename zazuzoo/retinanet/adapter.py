import os
from dl_to_csv import create_annotations_txt
from .retinanet_model import RetinaModel


class AdapterModel:

    def __init__(self, model_specs, hp_values):
        self.model_specs = model_specs
        self.annotation_type = model_specs['data']['annotation_type']
        self.hp_values = hp_values
        self.path = os.getcwd()
        self.output_path = os.path.join(self.path, 'output')

        self.classes_filepath = None
        self.annotations_train_filepath = None
        self.annotations_val_filepath = None
        self.coco_path = None
        if self.annotation_type == 'coco':
            self.coco_path = self.model_specs['data']['coco_path']
        elif self.annotation_type == 'csv' or self.annotation_type == 'dataloop':
            self.classes_filepath = os.path.join(self.output_path, 'classes.txt')
            self.annotations_train_filepath = os.path.join(self.output_path, 'annotations_train.txt')
            self.annotations_val_filepath = os.path.join(self.output_path, 'annotations_val.txt')
        self.retinanet_model = RetinaModel()

    def reformat(self):
        if self.annotation_type == 'coco':
            pass
        elif self.annotation_type == 'csv':
            pass
        elif self.annotation_type == 'dataloop':
            # convert dataloop annotations to csv styled annotations
            labels_list = self.model_specs['data']['labels_list']
            local_labels_path = os.path.join(self.path, self.model_specs['data']['labels_relative_path'])
            local_items_path = os.path.join(self.path, self.model_specs['data']['items_relative_path'])

            create_annotations_txt(annotations_path=local_labels_path,
                                   images_path=local_items_path,
                                   train_split=0.9,
                                   train_filepath=self.annotations_train_filepath,
                                   val_filepath=self.annotations_val_filepath,
                                   classes_filepath=self.classes_filepath,
                                   labels_list=labels_list)
            self.annotation_type == 'csv'

    def preprocess(self):
        self.retinanet_model.preprocess(dataset=self.annotation_type,
                                        csv_train=self.annotations_train_filepath,
                                        csv_val=self.annotations_val_filepath,
                                        csv_classes=self.classes_filepath,
                                        coco_path=self.coco_path,
                                        resize=self.hp_values['input_size'])

    def build(self):
        self.retinanet_model.build(depth=self.model_specs['training_configs']['depth'],
                                   learning_rate=self.hp_values['learning_rate'])

    def train(self):
        self.retinanet_model.train(epochs=self.model_specs['training_configs']['epochs'])

    def get_metrics(self):
        return {'val_accuracy': self.retinanet_model.get_metrics().item()}
