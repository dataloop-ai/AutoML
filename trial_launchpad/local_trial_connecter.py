import dtlpy as dl
import json
import torch
from logging_utils import logginger
from importlib import import_module
import threading


class TrialConnector():

    def __init__(self):
        self.logger = logginger(__name__)
        self.task_to_model = {'objectdetection': ('retinanet', 'yolov4', 'fasterrcnn', 'mobilenet')}
    # receives dict and saves checkpoint, adapter model only accepts checkpoints
    def run(self, inputs_dict):
        task = [key for key, val in self.task_to_model.items() if inputs_dict['model_specs']['name'] in val][0]
        Trial = getattr(import_module('.trial_adapter', task), 'TrialAdapter')
        checkpoint_path = 'meta_checkpoint.pt'
        device_index = inputs_dict['devices']['gpu_index']
        inputs_dict.pop('devices')
        torch.save(inputs_dict, checkpoint_path)
        trial = Trial(device_index)
        trial.load(checkpoint_path=checkpoint_path)
        self.logger.info('commencing training . . . ')
        trial.train()
        self.logger.info('training finished')
        meta_checkpoint = trial.get_checkpoint_metadata()
        if type(meta_checkpoint['metrics']['val_accuracy']) is not float:
            raise Exception(
                'trial, get_best_metrics method must return dict with only python floats. '
                'Not numpy floats or any other objects like that')
        return meta_checkpoint
