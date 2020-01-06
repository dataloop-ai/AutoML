import os
import threading
import logging
import torch
from plugins.local import LocalTrialConnector
from .thread_manager import ThreadManager
from zoo.convert2Yolo import convert
from main_pred import pred_run
from plugins import get_dataset_obj

logger = logging.getLogger('launcher')


class Launcher:
    def __init__(self, optimal_model, ongoing_trials=None, remote=False):
        self.optimal_model = optimal_model
        self.ongoing_trials = ongoing_trials
        self.remote = remote
        self.num_available_devices = torch.cuda.device_count()
        self.home_path = optimal_model.data['home_path']
        self.dataset_name = optimal_model.data['dataset_name']
        plugin_name = 'trainer' if self.ongoing_trials is None else 'trial'

        if self.optimal_model.name == 'yolov3':
            if self.optimal_model.data['annotation_type'] == 'coco':
                self._convert_coco_to_yolo_format()
                self.optimal_model.data['annotation_type'] = 'yolo'

        if self.remote:
            self.dataset_obj = get_dataset_obj()
            self.project = self.dataset_obj.project
            self._push_and_deploy_plugin(plugin_name=plugin_name)
        else:
            self.local_trial_connector = LocalTrialConnector(plugin_name)

    def predict(self, checkpoint_path):
        inputs = {
            'checkpoint_path': checkpoint_path,
            'name': self.optimal_model.name,
            'data': self.optimal_model.data
        }
        pred_run(checkpoint_path, self.optimal_model.name, self.home_path)

    def train_best_trial(self, best_trial):
        model_specs = self.optimal_model.unwrap()
        inputs = {
            'devices': {'gpu_index': 0},
            'hp_values': best_trial['hp_values'],
            'model_specs': model_specs,
        }
        if not self.remote:
            return self._run_demo_session(inputs)
        else:
            return self._run_remote_session(inputs)

    def launch_trials(self):
        if self.ongoing_trials is None:
            raise Exception('for this method ongoing_trials object must be passed during the init')
        if self.remote:
            self._launch_remote_trials()
        else:
            self._launch_local_trials()

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
            inputs = {
                'dataset_obj': self.dataset_obj,
                'hp_values': trial['hp_values'],
                'model_specs': model_specs
            }

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
        if not self.remote:
            metrics = self._run_demo_session(inputs)
        else:
            metrics = self._run_remote_session(inputs)
        results_dict[id_hash] = metrics
        logger.info('finshed thread: ' + thread_name)

    def _push_and_deploy_plugin(self, plugin_name):

        inputs = [
            {
                "type": "Dataset",
                "name": "dataset_obj"
            },
            {
                "type": "Json",
                "name": "model_specs"
            },
            {
                "type": "Json",
                "name": "hp_values"
            }]

        plugin = self.project.plugins.push(plugin_name=plugin_name,
                                           src_path=os.getcwd(),
                                           inputs=inputs)

        self.deployment = plugin.deployments.deploy(deployment_name=plugin.name,
                                                    plugin=plugin,
                                                    runtime={'gpu': True,
                                                             'numReplicas': 1,
                                                             'concurrency': 2,
                                                             'image': 'gcr.io/viewo-g/piper/agent/runner/gpu/main/zazu:latest'
                                                             },
                                                    bot=None)

    def _run_remote_session(self, inputs):

        metrics = self.deployment.sessions.create(deployment_id=self.deployment.id,
                                                  session_input=inputs,
                                                  sync=True).output
        return metrics

    def _run_demo_session(self, inputs):
        return self.local_trial_connector.run(inputs['devices'], inputs['model_specs'], inputs['hp_values'])
