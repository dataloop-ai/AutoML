import os
import json
import time
import threading
import logging
import torch
import glob
import shutil
from dataloop_services import LocalTrialConnector, LocalPredConnector
from .thread_manager import ThreadManager
from ObjectDetNet.convert2Yolo import convert
from dataloop_services.plugin_utils import get_dataset_obj
import dtlpy as dl
from logging_utils import logginger
from copy import deepcopy


logger = logginger(__name__)


class Launcher:
    def __init__(self, optimal_model, ongoing_trials=None, remote=False):
        self.optimal_model = optimal_model
        self.ongoing_trials = ongoing_trials
        self.remote = remote
        self.num_available_devices = torch.cuda.device_count()
        self.home_path = optimal_model.data['home_path']
        self.dataset_name = optimal_model.data['dataset_name']
        self.package_name = 'zazuml'
        if self.remote:
            dataset_obj = get_dataset_obj(optimal_model.dataloop)
            self.project = dl.projects.get(project_id=dataset_obj.projects[0])
            self.dataset_id = dataset_obj.id

            try:
                self.train_query = optimal_model.dataloop['train_query']
            except:
                self.train_query = dl.Filters().prepare()['filter']

            try:
                # TODO: TRAIN QUERY IS STILL BEING COPPIED
                try:
                    self.val_query = deepcopy(self.train_query)
                except:
                    self.val_query = dl.Filters().prepare()
                self.val_query['filter']['$and'][0]['dir'] = optimal_model.dataloop['test_dir']
            except:
                try:
                    self.val_query = optimal_model.dataloop['val_query']
                except:
                    self.val_query = dl.Filters().prepare()['filter']

            with open('global_configs.json', 'r') as fp:
                global_project_name = json.load(fp)['project']
            self.global_project = dl.projects.get(project_name=global_project_name)


        # TODO: dont convert here
        if self.optimal_model.name == 'yolov3':
            if self.optimal_model.data['annotation_type'] == 'coco':
                self._convert_coco_to_yolo_format()
                self.optimal_model.data['annotation_type'] = 'yolo'

    def predict(self, checkpoint_path):
        if self.remote:
            self._launch_predict_remote(checkpoint_path)
        else:
            self._launch_predict_local(checkpoint_path)

    def _launch_predict_local(self, checkpoint_path):
        self.local_pred_detector = LocalPredConnector()
        model_specs = self.optimal_model.unwrap()
        inputs = {'checkpoint_path': checkpoint_path,
                  'model_specs': model_specs}

        self._run_pred_demo_execution(inputs)

    def _launch_predict_remote(self, checkpoint_path):
        self.service = self.global_project.services.get(service_name='predict')
        model_specs = self.optimal_model.unwrap()
        dataset_input = dl.FunctionIO(type='Dataset', name='dataset', value={"dataset_id": self.dataset_id})
        checkpoint_path_input = dl.FunctionIO(type='Json', name='checkpoint_path', value={"checkpoint_path": checkpoint_path})
        val_query_input = dl.FunctionIO(type='Json', name='val_query', value=self.val_query)
        model_specs_input = dl.FunctionIO(type='Json', name='model_specs', value=model_specs)
        inputs = [dataset_input, val_query_input, checkpoint_path_input, model_specs_input]
        logger.info('checkpoint is type: ' + str(type(checkpoint_path)))
        try:
            logger.info("trying to get execution object")
            execution_obj = self._run_pred_remote_execution(inputs)
            logger.info("got execution object")
            # TODO: Turn execution_obj into metrics
            while execution_obj.latest_status['status'] != 'success':
                time.sleep(5)
                execution_obj = dl.executions.get(execution_id=execution_obj.id)
                if execution_obj.latest_status['status'] == 'failed':
                    raise Exception("plugin execution failed")
            logger.info("execution object status is successful")
            # download artifacts, should contain dir with txt file annotations
            # TODO: download many different metrics then should have id hash as well..
            self.project.artifacts.download(package_name=self.package_name,
                                            execution_id=execution_obj.id,
                                            local_path=os.getcwd())

        except Exception as e:
            Exception(' had an exception: \n', repr(e))

    def eval(self):
        pass

    def train_and_save_best_trial(self, best_trial, save_checkpoint_location):
        if self.remote:
            try:
                path_to_tensorboard_dir = 'runs'
                execution_obj = self._launch_remote_best_trial(best_trial)
                if os.path.exists(save_checkpoint_location):
                    logger.info('overwriting checkpoint.pt . . .')
                    os.remove(save_checkpoint_location)
                if os.path.exists(path_to_tensorboard_dir):
                    logger.info('overwriting tenorboards runs . . .')
                    os.rmdir(path_to_tensorboard_dir)
                # download artifacts, should contain checkpoint and tensorboard logs
                self.project.artifacts.download(package_name=self.package_name,
                                                execution_id=execution_obj.id,
                                                local_path=os.getcwd())
            except Exception as e:
                print(e)

        else:
            checkpoint = self._launch_local_best_trial(best_trial)
            if os.path.exists(save_checkpoint_location):
                logger.info('overwriting checkpoint.pt . . .')
                os.remove(save_checkpoint_location)
            torch.save(checkpoint, save_checkpoint_location)

    def launch_trials(self):
        if self.ongoing_trials is None:
            raise Exception('for this method ongoing_trials object must be passed during the init')
        if self.ongoing_trials.num_trials > 0:
            if self.remote:
                self._launch_remote_trials()

            else:
                self._launch_local_trials()

    def _launch_local_best_trial(self, best_trial):
        model_specs = self.optimal_model.unwrap()
        inputs = {
            'devices': {'gpu_index': 0},
            'hp_values': best_trial['hp_values'],
            'model_specs': model_specs,
        }

        return self._run_trial_demo_execution(inputs)

    def _launch_remote_best_trial(self, best_trial):
        model_specs = self.optimal_model.unwrap()
        dataset_input = dl.FunctionIO(type='Dataset', name='dataset', value={"dataset_id": self.dataset_id})
        train_query_input = dl.FunctionIO(type='Json', name='train_query', value=self.train_query)
        val_query_input = dl.FunctionIO(type='Json', name='val_query', value=self.val_query)
        hp_value_input = dl.FunctionIO(type='Json', name='hp_values', value=best_trial['hp_values'])
        model_specs_input = dl.FunctionIO(type='Json', name='model_specs', value=model_specs)
        inputs = [dataset_input, train_query_input, val_query_input, hp_value_input, model_specs_input]

        execution_obj = self._run_trial_remote_execution(inputs)
        while execution_obj.latest_status['status'] != 'success':
            time.sleep(5)
            execution_obj = dl.executions.get(execution_id=execution_obj.id)
            if execution_obj.latest_status['status'] == 'failed':
                raise Exception("package execution failed")
        return execution_obj

    def _launch_local_trials(self):
        self.local_trial_connector = LocalTrialConnector()
        threads = ThreadManager()
        model_specs = self.optimal_model.unwrap()
        logger.info('launching new set of trials')
        device = 0
        for trial_id, trial in self.ongoing_trials.trials.items():
            logger.info('launching trial_' + trial_id + ': ' + str(trial))
            inputs = {
                'devices': {'gpu_index': device},
                'hp_values': trial['hp_values'],
                'model_specs': model_specs
            }

            threads.new_thread(target=self._collect_metrics,
                               inputs=inputs,
                               trial_id=trial_id)
            device = device + 1

        threads.wait()
        ongoing_trials_results = threads.results
        for trial_id, metrics_and_checkpoint_dict in ongoing_trials_results.items():
            self.ongoing_trials.update_metrics(trial_id, metrics_and_checkpoint_dict)

    def _launch_remote_trials(self):
        self.service = self.global_project.services.get(service_name='trial')
        threads = ThreadManager()
        model_specs = self.optimal_model.unwrap()
        logger.info('launching new set of trials')
        for trial_id, trial in self.ongoing_trials.trials.items():
            dataset_input = dl.FunctionIO(type='Dataset', name='dataset', value={"dataset_id": self.dataset_id})
            train_query_input = dl.FunctionIO(type='Json', name='train_query', value=self.train_query)
            val_query_input = dl.FunctionIO(type='Json', name='val_query', value=self.val_query)
            hp_value_input = dl.FunctionIO(type='Json', name='hp_values', value=trial['hp_values'])
            model_specs_input = dl.FunctionIO(type='Json', name='model_specs', value=model_specs)
            inputs = [dataset_input, train_query_input, val_query_input, hp_value_input, model_specs_input]

            threads.new_thread(target=self._collect_metrics,
                               inputs=inputs,
                               trial_id=trial_id)

        threads.wait()
        ongoing_trials_results = threads.results
        for trial_id, metrics_and_checkpoint in ongoing_trials_results.items():
            self.ongoing_trials.update_metrics(trial_id, metrics_and_checkpoint)

    def _convert_coco_to_yolo_format(self):
        conversion_config_val = {
            "datasets": "COCO",
            "img_path": os.path.join(self.home_path, "images", "val" + self.dataset_name),
            "label": os.path.join(self.home_path, "annotations", "instances_val" + self.dataset_name + ".json"),
            "img_type": ".jpg",
            "manipast_path": os.path.join(self.home_path, "val_paths.txt"),
            "output_path": os.path.join(self.home_path, "labels", "val" + self.dataset_name),
            "cls_list": os.path.join(self.home_path, "d.names")
        }
        conversion_config_train = {
            "datasets": "COCO",
            "img_path": os.path.join(self.home_path, "images", "train" + self.dataset_name),
            "label": os.path.join(self.home_path, "annotations", "instances_train" + self.dataset_name + ".json"),
            "img_type": ".jpg",
            "manipast_path": os.path.join(self.home_path, "train_paths.txt"),
            "output_path": os.path.join(self.home_path, "labels", "train" + self.dataset_name),
            "cls_list": os.path.join(self.home_path, "d.names")
        }
        convert(conversion_config_val)
        convert(conversion_config_train)

    def _collect_metrics(self, inputs_dict, trial_id, results_dict):
        thread_name = threading.currentThread().getName()
        logger.info('starting thread: ' + thread_name)
        if self.remote:
            try:
                # checkpoint_path = 'best_' + trial_id + '.pt'
                checkpoint_path = 'checkpoint.pt'
                path_to_tensorboard_dir = 'runs'
                logger.info("trying to get execution objects")
                execution_obj = self._run_trial_remote_execution(inputs_dict)
                logger.info("got execution objects")
                # TODO: Turn execution_obj into metrics
                while execution_obj.latest_status['status'] != 'success':
                    time.sleep(5)
                    execution_obj = dl.executions.get(execution_id=execution_obj.id)
                    if execution_obj.latest_status['status'] == 'failed':
                        raise Exception("plugin execution failed")
                logger.info("execution object status is successful")
                if os.path.exists(checkpoint_path):
                    logger.info('overwriting checkpoint.pt . . .')
                    os.remove(checkpoint_path)
                if os.path.exists(path_to_tensorboard_dir):
                    logger.info('overwriting tenorboards runs . . .')
                    shutil.rmtree(path_to_tensorboard_dir)
                # download artifacts, should contain metrics and tensorboard runs
                # TODO: download many different metrics then should have id hash as well..
                self.project.artifacts.download(package_name=self.package_name,
                                                execution_id=execution_obj.id,
                                                local_path=os.getcwd())
                logger.info('going to load ' + checkpoint_path + ' into checkpoint')
                if torch.cuda.is_available():
                    checkpoint = torch.load(checkpoint_path)
                else:
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                os.remove(checkpoint_path)

            except Exception as e:
                Exception('The thread ' + thread_name + ' had an exception: \n', repr(e))
        else:
            checkpoint = self._run_trial_demo_execution(inputs_dict)

        results_dict[trial_id] = {'metrics': checkpoint['metrics'],
                'checkpoint': checkpoint}
        logger.info('finished thread: ' + thread_name)


    def _run_trial_remote_execution(self, inputs):
        logger.info('running new execution . . .')

        execution_obj = self.service.execute(execution_input=inputs, function_name='run')
        logger.info('executing: ' + execution_obj.id)
        return execution_obj

    def _run_pred_remote_execution(self, inputs):
        logger.info('running new execution . . .')

        execution_obj = self.service.execute(execution_input=inputs, function_name='run')
        logger.info('executing: ' + execution_obj.id)
        return execution_obj

    def _run_trial_demo_execution(self, inputs_dict):
        return self.local_trial_connector.run(inputs_dict)

    def _run_pred_demo_execution(self, inputs):
        return self.local_pred_detector.run(inputs['checkpoint_path'], inputs['model_specs'])
