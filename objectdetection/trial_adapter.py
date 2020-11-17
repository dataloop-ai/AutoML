import os
# sys.path.insert(1, os.path.dirname(__file__))
from .model_trainer import ModelTrainer
from .predict import detect, detect_single_image

from copy import deepcopy
import random
import time
import hashlib
import torch
import logging

logger = logging.getLogger(__name__)


def generate_trial_id():
    s = str(time.time()) + str(random.randint(1, 1e7))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]


class TrialAdapter(ModelTrainer):

    def __init__(self):
        pass

    def load(self, checkpoint_path='checkpoint.pt'):
        # the only necessary keys for load are ['devices', 'model_specs']
        trial_checkpoint = torch.load(checkpoint_path)

        try:
            hp_values = trial_checkpoint['hp_values']
        except:
            hp_values = {}
        checkpoint = None
        if 'model' in trial_checkpoint.keys():
            checkpoint = deepcopy(trial_checkpoint)
            for x in ['devices', 'model_specs', 'hp_values']:
                checkpoint.pop(x)
            epoch = checkpoint['epoch']
            hp_values['hyperparameter_tuner/initial_epoch'] = epoch

        self.devices = trial_checkpoint['devices']
        self.model_specs = trial_checkpoint['model_specs']
        self.hp_values = hp_values

        self.annotation_type = self.model_specs['data']['annotation_type']
        self.path = os.getcwd()
        self.output_path = os.path.join(self.path, 'output')

        self.model_specs['training_configs'].update(hp_values)
        self.configs = self.model_specs['training_configs']

        self.classes_filepath = None
        self.annotations_train_filepath = None
        self.annotations_val_filepath = None
        self.home_path = None
        try:
            past_trial_id = self.configs['hyperparameter_tuner/past_trial_id']
        except:
            past_trial_id = None
        try:
            new_trial_id = self.configs['hyperparameter_tuner/new_trial_id']
        except Exception as e:
            raise Exception('make sure a new trial id was passed, got this error: ' + repr(e))
        if 'hyperparameter_tuner/initial_epoch' not in self.configs.keys():
            self.configs['hyperparameter_tuner/initial_epoch'] = 0

        if self.annotation_type == 'coco':
            self.home_path = self.model_specs['data']['home_path']
            self.dataset_name = self.model_specs['data']['dataset_name']

        super().__init__(self.devices['gpu_index'], self.home_path, new_trial_id, past_trial_id, checkpoint)

    def train(self):
        super().preprocess(augment_policy=self.configs['augment_policy'],
                           dataset=self.annotation_type,
                           csv_train=self.annotations_train_filepath,
                           csv_val=self.annotations_val_filepath,
                           csv_classes=self.classes_filepath,
                           train_set_name='train' + self.dataset_name,
                           val_set_name='val' + self.dataset_name,
                           resize=self.configs['input_size'],
                           batch=self.configs['batch'])

        super().build(model_name=self.model_specs['name'],
                      depth=self.configs['depth'],
                      learning_rate=self.configs['learning_rate'],
                      ratios=self.configs['anchor_ratios'],
                      scales=self.configs['anchor_scales'])

        super().train(epochs=self.configs['hyperparameter_tuner/epochs'],
                      init_epoch=self.configs['hyperparameter_tuner/initial_epoch'])

    def get_checkpoint_metadata(self):
        logger.info('getting best checkpoint')
        checkpoint = super().get_best_checkpoint()
        logging.info('got best checkpoint')
        checkpoint['hp_values'] = self.hp_values
        checkpoint['model_specs'] = self.model_specs
        checkpoint['devices'] = self.devices
        checkpoint['checkpoint_path'] = super().save_best_checkpoint_path
        checkpoint.pop('model')
        logging.info('checkpoint keys: ' + str(checkpoint.keys()))
        return checkpoint

    @property
    def checkpoint_path(self):
        return super().save_best_checkpoint_path

    def load_inference(self, checkpoint_path):
        if torch.cuda.is_available():
            logger.info('cuda available')
            self.inference_checkpoint = torch.load(checkpoint_path)
        else:
            logger.info('run on cpu')
            self.inference_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        return self.inference_checkpoint

    def predict_single_image(self, image_path, checkpoint_path='checkpoint.pt'):
        if hasattr(self, 'inference_checkpoint'):
            return detect_single_image(self.inference_checkpoint, image_path)
        else:
            self.load_inference(checkpoint_path)
            return detect_single_image(self.inference_checkpoint, image_path)

    def predict_items(self, items, checkpoint_path, with_upload=True, model_name='object_detection'):
        for item in items:
            dirname = self.predict_item(item, checkpoint_path, with_upload, model_name)

        return dirname

    def delete_stuff(self):
        super().delete_stuff()
