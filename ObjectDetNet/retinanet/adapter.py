import os

import sys
sys.path.insert(1, os.path.dirname(__file__))
from retinanet_model import RetinaModel
from predict import detect, detect_single_image

from copy import deepcopy
import random
import time
import hashlib
import json
import torch
import dtlpy as dl
import logging
logger = logging.getLogger(__name__)


#def combine_values(configs_under, configs_over):
#    for hp in configs_over.keys():
#        configs_under[hp] = configs_over[hp]

#   return configs_under

def generate_trial_id():
    s = str(time.time()) + str(random.randint(1, 1e7))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]


class AdapterModel:

    def load_from_checkpoint(self, model_id, checkpoint_id):
        model = dl.models.get(model_id=model_id)
        checkpoint = model.checkpoints.get(checkpoint_id=checkpoint_id)
        checkpoint_path = checkpoint.download(local_path=os.getcwd())
        self.load(checkpoint_path=checkpoint_path)
        self.model = model

    def load(self, checkpoint_path='checkpoint.pt'):
        # the only necessary keys for load are ['devices', 'model_specs']
        trial_checkpoint = torch.load(checkpoint_path)

        devices = trial_checkpoint['devices']
        model_specs = trial_checkpoint['model_specs']
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
            hp_values['tuner/initial_epoch'] = epoch

        self.devices = devices
        self.model_specs = model_specs
        self.hp_values = hp_values
     
        self.annotation_type = model_specs['data']['annotation_type']
        self.path = os.getcwd()
        self.output_path = os.path.join(self.path, 'output')
        
        # unify training configs and hp_values using dict.update
        #self.configs = combine_values(self.model_specs['training_configs'], hp_values) 
        self.model_specs['training_configs'].update(hp_values)
        self.configs = self.model_specs['training_configs']
        
        self.classes_filepath = None
        self.annotations_train_filepath = None
        self.annotations_val_filepath = None
        self.home_path = None
        try:
            past_trial_id = self.configs['tuner/past_trial_id']
        except:
            past_trial_id = None
        try:
            new_trial_id = self.configs['tuner/new_trial_id']
        except Exception as e:
            raise Exception('make sure a new trial id was passed, got this error: ' + repr(e))
        if 'tuner/initial_epoch' not in self.configs.keys():
            self.configs['tuner/initial_epoch'] = 0

        if self.annotation_type == 'coco':
            self.home_path = self.model_specs['data']['home_path']
            self.dataset_name = self.model_specs['data']['dataset_name']
        elif self.annotation_type == 'csv' or self.annotation_type == 'dataloop':
            self.classes_filepath = os.path.join(self.output_path, 'classes.txt')
            self.annotations_train_filepath = os.path.join(self.output_path, 'annotations_train.txt')
            self.annotations_val_filepath = os.path.join(self.output_path, 'annotations_val.txt')
        self.retinanet_model = RetinaModel(devices['gpu_index'], self.home_path, new_trial_id, past_trial_id,
                                           checkpoint)

    def reformat(self):
        pass
        # if self.annotation_type == 'coco':
        #     pass
        # elif self.annotation_type == 'csv':
        #     pass
        # elif self.annotation_type == 'dataloop':
        #     # convert dataloop annotations to csv styled annotations
        #     labels_list = self.model_specs['data']['labels_list']
        #     local_labels_path = os.path.join(self.path, self.model_specs['data']['labels_relative_path'])
        #     local_items_path = os.path.join(self.path, self.model_specs['data']['items_relative_path'])
        #
        #     create_annotations_txt(annotations_path=local_labels_path,
        #                            images_path=local_items_path,
        #                            train_split=0.9,
        #                            train_filepath=self.annotations_train_filepath,
        #                            val_filepath=self.annotations_val_filepath,
        #                            classes_filepath=self.classes_filepath,
        #                            labels_list=labels_list)
        #     self.annotation_type == 'csv'

    def preprocess(self):
        self.retinanet_model.preprocess(dataset=self.annotation_type,
                                        csv_train=self.annotations_train_filepath,
                                        csv_val=self.annotations_val_filepath,
                                        csv_classes=self.classes_filepath,
                                        coco_path=True,
                                        train_set_name='train' + self.dataset_name,
                                        val_set_name='val' + self.dataset_name,
                                        resize=self.configs['input_size'])

    def build(self):
        self.retinanet_model.build(depth=self.configs['depth'],
                                   learning_rate=self.configs['learning_rate'],
                                   ratios=self.configs['anchor_ratios'],
                                   scales=self.configs['anchor_scales'])

    def train(self):
        self.retinanet_model.train(epochs=self.configs['tuner/epochs'],
                                   init_epoch=self.configs['tuner/initial_epoch'])

    def get_checkpoint(self):
        logger.info('getting best checkpoint')
        checkpoint = self.retinanet_model.get_best_checkpoint()
        logging.info('got best checkpoint')
        checkpoint['hp_values'] = self.hp_values
        checkpoint['model_specs'] = self.model_specs
        checkpoint['devices'] = self.devices
        logging.info('checkpoint keys: ' + str(checkpoint.keys()))
        return checkpoint

    @property
    def checkpoint_path(self):
        return self.retinanet_model.save_best_checkpoint_path

    def save(self, save_path='checkpoint.pt'):
        checkpoint = self.get_checkpoint()
        torch.save(checkpoint, save_path)
        return save_path

    def upload_checkpoint(self, model_id=None, checkpoint_name='checkpoint'):
        if model_id:
            model = dl.models.get(model_id=model_id)
            self.model = model
        save_path = os.path.join(os.getcwd(), checkpoint_name + '.pt')
        self.save(save_path)
        self.model.checkpoints.upload(checkpoint_name=checkpoint_name, local_path=save_path)

    def predict(self, output_dir='checkpoint0', checkpoint_path='checkpoint.pt', home_path=None):
        try:
            return detect(self.inference_checkpoint, output_dir, home_path=home_path, visualize=True)
        except:
            try:
                self.load_inference(checkpoint_path)
                return detect(checkpoint=self.inference_checkpoint, output_dir=output_dir, home_path=home_path, visualize=True)
            except:
                checkpoint = self.get_checkpoint()
                return detect(checkpoint, output_dir)

    def load_inference(self, checkpoint_path):
        if torch.cuda.is_available():
            logger.info('cuda available')
            self.inference_checkpoint = torch.load(checkpoint_path)
        else:
            logger.info('run on cpu')
            self.inference_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        return self.inference_checkpoint

    def load_from_inference_checkpoint(self, model_id, checkpoint_id):
        model = dl.models.get(model_id=model_id)
        checkpoint = model.checkpoints.get(checkpoint_id=checkpoint_id)
        checkpoint_path = checkpoint.download(local_path=os.getcwd())
        self.load_inference(checkpoint_path=checkpoint_path)
        self.model = model

    def predict_single_image(self, image_path, checkpoint_path='checkpoint.pt'):
        if hasattr(self, 'inference_checkpoint'):
            return detect_single_image(self.inference_checkpoint, image_path)
        else:
            self.load_inference(checkpoint_path)
            return detect_single_image(self.inference_checkpoint, image_path)

    def predict_item(self, item, checkpoint_path=None, with_upload=True, model_name='retinanet'):
        filepath = item.download()
        results_path = self.predict_single_image(image_path=filepath, checkpoint_path=checkpoint_path)
        if with_upload:
            with open(results_path) as fg:
                results = fg.readlines()
            builder = item.annotations.builder()
            for result in results:
                result_ls = result.split(' ')
                builder.add(dl.Box(left=int(result_ls[2]), top=int(result_ls[3]), right=int(result_ls[4]),
                                    bottom=int(result_ls[5]), label=result_ls[0]),
                            model_info={'confidence': result_ls[1], 'name': model_name})
            item.annotations.upload(builder)

        dirname = os.path.dirname(filepath)
        return dirname

    def predict_items(self, items, checkpoint_path, with_upload=True, model_name='retinanet'):
        for item in items:
            dirname = self.predict_item(item, checkpoint_path, with_upload, model_name)

        return dirname

    def delete_stuff(self):
        self.retinanet_model.delete_stuff()
