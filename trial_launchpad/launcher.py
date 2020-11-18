import os
import json
import time
import threading
import logging
import torch
import glob
import shutil
from .local_trial_connecter import TrialConnector
from .thread_manager import ThreadManager
from dataloop_services.plugin_utils import get_dataset_obj
import dtlpy as dl
from logging_utils import logginger
from copy import deepcopy


logger = logginger(__name__)


class Launcher:
    def __init__(self, ongoing_trials, model_fn, training_configs, home_path, annotation_type):
        self.ongoing_trials = ongoing_trials
        self.num_available_devices = torch.cuda.device_count()
        self.home_path = home_path
        self.annotation_type = annotation_type
        self.model_fn = model_fn
        self.training_configs = training_configs

    def launch_trial(self, hp_values):

        inputs = {
            'devices': {'gpu_index': 0},
            'hp_values': hp_values,
            'model_fn': self.model_fn,
            'training_configs': self.training_configs,
            'home_path': self.home_path,
            'annotation_type': self.annotation_type
        }

        meta_checkpoint = self.trial_connector.run(inputs)
        return {'metrics': meta_checkpoint['metrics'],
                'meta_checkpoint': meta_checkpoint}

    def launch_trials(self):
        if self.ongoing_trials is None:
            raise Exception('for this method ongoing_trials object must be passed during the init')
        if self.ongoing_trials.num_trials > 0:
            self.trial_connector = TrialConnector()
            threads = ThreadManager()
            logger.info('launching new set of trials')
            device = 0
            for trial_id, trial in self.ongoing_trials.trials.items():
                logger.info('launching trial_' + trial_id + ': ' + str(trial))
                inputs = {
                    'devices': {'gpu_index': device},
                    'hp_values': trial['hp_values'],
                    'model_fn': self.model_fn,
                    'training_configs': self.training_configs,
                    'home_path': self.home_path,
                    'annotation_type': self.annotation_type
                }

                threads.new_thread(target=self._collect_metrics,
                                   inputs=inputs,
                                   trial_id=trial_id)
                device = device + 1

            threads.wait()
            ongoing_trials_results = threads.results
            for trial_id, metrics_and_checkpoint_dict in ongoing_trials_results.items():
                self.ongoing_trials.update_metrics(trial_id, metrics_and_checkpoint_dict)

    def _collect_metrics(self, inputs_dict, trial_id, results_dict):
        thread_name = threading.currentThread().getName()
        logger.info('starting thread: ' + thread_name)
        meta_checkpoint = self.trial_connector.run(inputs_dict)
        results_dict[trial_id] = {'metrics': meta_checkpoint['metrics'],
                'meta_checkpoint': meta_checkpoint}
        logger.info('finished thread: ' + thread_name)


