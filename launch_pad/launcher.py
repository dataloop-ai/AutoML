import os
import json
import time
import threading
import logging
import torch
from local_plugin import LocalTrialConnector
from .thread_manager import ThreadManager
from zoo.convert2Yolo import convert
from main_pred import pred_run
from plugin_utils import get_dataset_obj
import dtlpy as dl

logger = logging.getLogger('launcher')


class Launcher:
    def __init__(self, optimal_model, ongoing_trials=None, remote=False):
        self.optimal_model = optimal_model
        self.ongoing_trials = ongoing_trials
        self.remote = remote
        self.num_available_devices = torch.cuda.device_count()
        self.home_path = optimal_model.data['home_path']
        self.dataset_name = optimal_model.data['dataset_name']
        self.package_name = 'trainer' if self.ongoing_trials is None else 'trial'

        if self.optimal_model.name == 'yolov3':
            if self.optimal_model.data['annotation_type'] == 'coco':
                self._convert_coco_to_yolo_format()
                self.optimal_model.data['annotation_type'] = 'yolo'

        if self.remote:
            self.dataset_obj = get_dataset_obj()
            self.dataset_id = self.dataset_obj.id
            self.project = self.dataset_obj.project
            self._push_and_deploy_package(package_name=self.package_name)
        else:
            self.local_trial_connector = LocalTrialConnector(self.package_name)

    def predict(self, checkpoint_path):
        pred_run(checkpoint_path, self.optimal_model.name, self.home_path)

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
                for artifact in self.project.artifacts.list(package_name=self.package_name,
                                                            execution_id=execution_obj.id):
                    artifact.download(local_path=os.getcwd())
            except Exception as e:
                print(e)

            self.service.delete()
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
                self.service.delete()
            else:
                self._launch_local_trials()

    def _launch_local_best_trial(self, best_trial):
        model_specs = self.optimal_model.unwrap()
        inputs = {
            'devices': {'gpu_index': 0},
            'hp_values': best_trial['hp_values'],
            'model_specs': model_specs,
        }

        return self._run_demo_execution(inputs)

    def _launch_remote_best_trial(self, best_trial):
        model_specs = self.optimal_model.unwrap()
        dataset_input = dl.FunctionIO(type='Dataset', name='dataset', value={"dataset_id": self.dataset_id})
        hp_value_input = dl.FunctionIO(type='Json', name='hp_values', value=best_trial['hp_values'])
        model_specs_input = dl.FunctionIO(type='Json', name='model_specs', value=model_specs)
        inputs = [dataset_input, hp_value_input, model_specs_input]

        execution_obj = self._run_remote_execution(inputs)
        while execution_obj.latest_status['status'] != 'success':
            time.sleep(5)
            execution_obj = dl.executions.get(execution_id=execution_obj.id)
            if execution_obj.latest_status['status'] == 'failed':
                raise Exception("package execution failed")
        return execution_obj

    def _launch_local_trials(self):
        threads = ThreadManager()
        model_specs = self.optimal_model.unwrap()
        logger.info('launching new set of trials')
        device = 0
        for trial_id, trial in self.ongoing_trials.trials.items():
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
        for trial_id, metrics in ongoing_trials_results.items():
            self.ongoing_trials.update_metrics(trial_id, metrics)

    def _launch_remote_trials(self):
        threads = ThreadManager()
        model_specs = self.optimal_model.unwrap()
        logger.info('launching new set of trials')
        for trial_id, trial in self.ongoing_trials.trials.items():
            dataset_input = dl.FunctionIO(type='Dataset', name='dataset', value={"dataset_id": self.dataset_id})
            hp_value_input = dl.FunctionIO(type='Json', name='hp_values', value=trial['hp_values'])
            model_specs_input = dl.FunctionIO(type='Json', name='model_specs', value=model_specs)
            inputs = [dataset_input, hp_value_input, model_specs_input]

            threads.new_thread(target=self._collect_metrics,
                               inputs=inputs,
                               trial_id=trial_id)

        threads.wait()
        ongoing_trials_results = threads.results
        for trial_id, metrics in ongoing_trials_results.items():
            self.ongoing_trials.update_metrics(trial_id, metrics)

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

    def _collect_metrics(self, inputs, id_hash, results_dict):
        thread_name = threading.currentThread().getName()
        logger.info('starting thread: ' + thread_name)
        if self.remote:
            try:
                metrics_path = 'metrics.json'
                path_to_tensorboard_dir = 'runs'
                execution_obj = self._run_remote_execution(inputs)
                # TODO: Turn execution_obj into metrics
                while execution_obj.latest_status['status'] != 'success':
                    time.sleep(5)
                    execution_obj = dl.executions.get(execution_id=execution_obj.id)
                    if execution_obj.latest_status['status'] == 'failed':
                        raise Exception("plugin execution failed")

                if os.path.exists(metrics_path):
                    logger.info('overwriting checkpoint.pt . . .')
                    os.remove(metrics_path)
                if os.path.exists(path_to_tensorboard_dir):
                    logger.info('overwriting tenorboards runs . . .')
                    os.rmdir(path_to_tensorboard_dir)
                # download artifacts, should contain metrics and tensorboard runs
                for artifact in self.project.artifacts.list(package_name=self.package_name,
                                                            execution_id=execution_obj.id):
                    artifact.download(local_path=os.getcwd())

                with open(metrics_path, 'r') as fp:
                    metrics = json.load(fp)
                os.remove(metrics_path)
            except Exception as e:
                print('The thread ' + thread_name + ' had an exception: \n', e)
        else:
            metrics = self._run_demo_execution(inputs)

        results_dict[id_hash] = metrics
        logger.info('finshed thread: ' + thread_name)

    def _push_and_deploy_package(self, package_name):
        logger.info('dtlpy version:', dl.__version__)
        dataset_input = dl.FunctionIO(type='Dataset', name='dataset')
        hp_value_input = dl.FunctionIO(type='Json', name='hp_values')
        model_specs_input = dl.FunctionIO(type='Json', name='model_specs')
        input_to_init = dl.FunctionIO(type='Json', name='package_name', value=package_name)

        inputs = [dataset_input, hp_value_input, model_specs_input]
        function = dl.PackageFunction(name='run', inputs=inputs, outputs=[], description='')
        module = dl.PackageModule(entry_point='service_executor.py', name='service_executor', functions=[function],
                                  init_inputs=[input_to_init])

        package = self.project.packages.push(
            package_name=package_name,
            src_path=os.getcwd(),
            modules=[module])

        logger.info('deploying package . . .')
        self.service = package.services.deploy(service_name=package.name,
                                               module_name='service_executor',
                                               package=package,
                                               runtime={'gpu': True,
                                                        'numReplicas': 1,
                                                        'concurrency': 2,
                                                        'runnerImage': 'buffalonoam/zazu-image:0.2'
                                                        })

    def _run_remote_execution(self, inputs):
        logger.info('running new execution . . .')

        execution_obj = self.service.invoke(execution_input=inputs, function_name='run')
        return execution_obj

    def _run_demo_execution(self, inputs):
        return self.local_trial_connector.run(inputs['devices'], inputs['model_specs'], inputs['hp_values'])
