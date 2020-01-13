import os
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
        plugin_name = 'trainer' if self.ongoing_trials is None else 'trial'

        if self.optimal_model.name == 'yolov3':
            if self.optimal_model.data['annotation_type'] == 'coco':
                self._convert_coco_to_yolo_format()
                self.optimal_model.data['annotation_type'] = 'yolo'

        if self.remote:
            self.dataset_obj = get_dataset_obj()
            self.dataset_id = self.dataset_obj.id
            self.project = self.dataset_obj.project
            self._push_and_deploy_plugin(plugin_name=plugin_name)
            # self.deployment = self.project.deployments.get(deployment_name='trial')
            # plugin = self.project.plugins.get(plugin_id=self.deployment.pluginId)
            # print(plugin)
            # print('***')
        else:
            self.local_trial_connector = LocalTrialConnector(plugin_name)

    def predict(self, checkpoint_path):
        pred_run(checkpoint_path, self.optimal_model.name, self.home_path)

    def train_best_trial(self, best_trial):
        if self.remote:
            return self._launch_remote_best_trial(best_trial)
        else:
            return self._launch_local_best_trial(best_trial)

    def launch_trials(self):
        if self.ongoing_trials is None:
            raise Exception('for this method ongoing_trials object must be passed during the init')
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

        return self._run_demo_session(inputs)

    def _launch_remote_best_trial(self, best_trial):
        model_specs = self.optimal_model.unwrap()
        dataset_input = dl.PluginInput(type='Dataset', name='dataset', value={"dataset_id": self.dataset_id})
        hp_value_input = dl.PluginInput(type='Json', name='hp_values', value=best_trial['hp_values'])
        model_specs_input = dl.PluginInput(type='Json', name='model_specs', value=model_specs)
        inputs = [dataset_input, hp_value_input, model_specs_input]

        session_obj = self._run_remote_session(inputs)
        # TODO: Turn session_obj into checkpoint
        session_obj



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
            dataset_input = dl.PluginInput(type='Dataset', name='dataset', value={"dataset_id": self.dataset_id})
            hp_value_input = dl.PluginInput(type='Json', name='hp_values', value=trial['hp_values'])
            model_specs_input = dl.PluginInput(type='Json', name='model_specs', value=model_specs)
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
        if not self.remote:
            metrics = self._run_demo_session(inputs)
        else:
            session_obj = self._run_remote_session(inputs)
            # TODO: Turn session_obj into metrics
        results_dict[id_hash] = metrics
        logger.info('finshed thread: ' + thread_name)

    def _push_and_deploy_plugin(self, plugin_name):

        dataset_input = dl.PluginInput(type='Dataset', name='dataset')
        print('dtlpy version:', dl.__version__)
        hp_value_input = dl.PluginInput(type='Json', name='hp_values')
        model_specs_input = dl.PluginInput(type='Json', name='model_specs')

        inputs = [dataset_input, hp_value_input, model_specs_input]

        plugin = self.project.plugins.push(plugin_name=plugin_name,
                                           src_path=os.getcwd(),
                                           inputs=inputs)

        self.deployment = plugin.deployments.deploy(deployment_name=plugin.name,
                                                    plugin=plugin,
                                                    runtime={'gpu': True,
                                                             'numReplicas': 1,
                                                             'concurrency': 2,
                                                             'runnerImage': 'buffalonoam/zazu-image:0.2'
                                                             },
                                                    bot=None,
                                                    config={'plugin_name': plugin.name})

    def _run_remote_session(self, inputs):

        session_obj = self.deployment.sessions.create(deployment_id=self.deployment.id,
                                                      session_input=inputs)
        return session_obj

    def _run_demo_session(self, inputs):
        return self.local_trial_connector.run(inputs['devices'], inputs['model_specs'], inputs['hp_values'])
