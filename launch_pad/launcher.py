import json
import os
import threading
import logging
import torch
from main import PluginRunner
from threading import Thread
from .thread_manager import ThreadManager
from zazoo.convert2Yolo import convert
logger = logging.getLogger('launcher')


class Launcher:
    def __init__(self, optimal_model, ongoing_trials, remote=False):
        self.optimal_model = optimal_model
        self.ongoing_trials = ongoing_trials
        self.remote = remote
        self.num_available_devices = torch.cuda.device_count()
        self.home_path = optimal_model.data['home_path']
        self.dataset_name = optimal_model.data['dataset_name']
        if self.optimal_model.name == 'yolov3':
            if self.optimal_model.data['annotation_type'] == 'coco':
                self._convert_coco_to_yolo_format()

        if self.remote:
            self._push_and_deploy_plugin()
        else:
            self.plugin = PluginRunner()

    def launch_c(self):
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

    def _push_and_deploy_plugin(self):
        import dtlpy as dl
        dl.setenv('dev')
        project = dl.projects.get(project_id="fcdd792b-5146-4c62-8b27-029564f1b74e")
        plugin = project.plugins.push(src_path='/Users/noam/zazuML')
        self.deployment = plugin.deployments.deploy(deployment_name='thisdeployment',
                                                    plugin=plugin,
                                                    config={
                                                        'project_id': project.id,
                                                        'plugin_name': plugin.name
                                                    },
                                                    runtime={
                                                        'gpu': True,
                                                        'numReplicas': 1,
                                                        'concurrency': 1,
                                                        'image': 'gcr.io/viewo-g/piper/custom/zazuim:1.0'
                                                    })

    def _run_remote_session(self, inputs):

        # deployment = dl.projects.get(project_id="fcdd792b-5146-4c62-8b27-029564f1b74e").deployments.get(deployment_name="thisdeployment")

        metrics = self.deployment.sessions.create(deployment_id=self.deployment.id,
                                                  session_input=inputs,
                                                  sync=True).output
        return metrics

    @staticmethod
    def _run_local_session(inputs):

        dict_keys = inputs.keys()
        mock = {"inputs": [], "config": {}}
        for key in dict_keys:
            mock['inputs'].append({"name": key, "value": inputs[key]})

        with open('mock.json', 'w') as f:
            json.dump(mock, f)
        metrics = dl.plugins.test_local_plugin(os.getcwd())
        return metrics

    def _run_demo_session(self, inputs):
        return self.plugin.run(inputs['devices'], inputs['model_specs'], inputs['hp_values'])
