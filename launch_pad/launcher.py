import dtlpy as dl
import json
import os
from main import PluginRunner


class Launcher:
    def __init__(self, optimal_model, ongoing_trials, remote=False):
        self.optimal_model = optimal_model
        self.ongoing_trials = ongoing_trials
        self.remote = remote

        if self.remote == 1:
            self._push_and_deploy_plugin()
        elif self.remote == -1:
            self.plugin = PluginRunner()

    def launch_c(self):
        for trial_id, trial in self.ongoing_trials.trials.items():
            inputs = {
                'hp_values': trial['hp_values'],
                'model_specs': self.optimal_model.unwrap()
            }

            if self.remote == 1:
                metrics = self._run_remote_session(inputs)
            elif self.remote == 0:
                metrics = self._run_local_session(inputs)
            else:
                metrics = self._run_demo_session(inputs)
            self.ongoing_trials.update_metrics(trial_id, metrics)

    def _push_and_deploy_plugin(self):
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
        return self.plugin.run(inputs['model_specs'], inputs['hp_values'])
