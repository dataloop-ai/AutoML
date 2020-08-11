from model_selector import find_model
from launch_pad import Launcher
from hyperparameter_tuner import Tuner, OngoingTrials
from spec import ConfigSpec, OptModel
from spec import ModelsSpec
from logging_utils import init_logging, logginger
from dataloop_services import deploy_model, deploy_zazu, push_package, update_service, get_dataset_obj, deploy_predict, \
    deploy_zazu_timer
import argparse
import os
import torch
import json
import dtlpy as dl

logger = logginger(__name__)


class ZaZu:
    def __init__(self, opt_model, remote=False):
        self.remote = remote
        self.opt_model = opt_model
        self.path_to_most_suitable_model = 'model.txt'
        self.path_to_best_trial = 'best_trial.json'
        self.path_to_trials = 'trials.json'
        self.path_to_best_checkpoint = 'checkpoint.pt'
        models_spec_path = 'models.json'
        self.models = ModelsSpec(models_spec_path)

    def find_best_model(self):
        closest_model = find_model(self.opt_model, self.models)
        logger.info(str(closest_model))

        if os.path.exists(self.path_to_most_suitable_model):
            logger.info('overwriting model.txt . . .')
            os.remove(self.path_to_most_suitable_model)
        with open(self.path_to_most_suitable_model, "w") as f:
            f.write(closest_model)
        self.update_optimal_model()

    def hp_search(self):
        if not self.remote:
            if self.opt_model.max_instances_at_once > torch.cuda.device_count():
                print(torch.cuda.is_available())
                raise Exception(''' 'max_instances_at_once' must be smaller or equal to the number of available gpus''')
        if not hasattr(self.opt_model, 'name'):
            logger.info("no 'update_optimal_model' method, checking for model.txt file . . . ")
            self.update_optimal_model()
        # initialize hyperparameter_tuner and gun i.e.
        ongoing_trials = OngoingTrials()
        tuner = Tuner(self.opt_model, ongoing_trials)
        gun = Launcher(self.opt_model, ongoing_trials, remote=self.remote)
        logger.info('commencing hyper-parameter search . . . ')
        tuner.search_hp()
        gun.launch_trials()
        tuner.end_trial()
        # starting second set of trials
        tuner.search_hp()
        while ongoing_trials.status is not 'STOPPED':
            gun.launch_trials()
            tuner.end_trial()
            # starting next set of trials
            tuner.search_hp()

        trials = tuner.get_trials()
        sorted_trial_ids = tuner.get_sorted_trial_ids()

        string1 = self.path_to_best_checkpoint.split('.')[0]
        for i in range(len(sorted_trial_ids)):
            save_checkpoint_location = string1 + str(i) + '.pt'
            if os.path.exists(save_checkpoint_location):
                logger.info('overwriting checkpoint . . .')
                os.remove(save_checkpoint_location)
            logger.info('trial ' + sorted_trial_ids[i] + '\tval: ' + str(trials[sorted_trial_ids[i]]['metrics']))
            torch.save(trials[sorted_trial_ids[i]]['checkpoint'], save_checkpoint_location)

        logger.info('best trial: ' + str(trials[sorted_trial_ids[0]]['hp_values']) + '\nbest value: ' + str(
            trials[sorted_trial_ids[0]]['metrics']))

        best_trial = trials[sorted_trial_ids[0]]['hp_values']
        if os.path.exists(self.path_to_best_trial):
            logger.info('overwriting best_trial.json . . .')
            os.remove(self.path_to_best_trial)
        with open(self.path_to_best_trial, 'w') as fp:
            json.dump(best_trial, fp)
            logger.info('results saved to best_trial.json')

    def train_new_model(self):
        # to train a new model you must have updated the found model and the best trial
        if not hasattr(self.opt_model, 'name'):
            logger.info("no 'update_optimal_model' method, checking for model.txt file . . . ")
            self.update_optimal_model()
        if not os.path.exists(self.path_to_best_trial):
            raise Exception('''best_trial.json doesn't exist, you can run "hp_search" to get it''')
        with open(self.path_to_best_trial, 'r') as fp:
            best_trial = json.load(fp)

        gun = Launcher(self.opt_model, remote=self.remote)
        gun.train_and_save_best_trial(best_trial, self.path_to_best_checkpoint)

    def update_optimal_model(self):
        # this will update opt_model with chosen model
        if not os.path.exists(self.path_to_most_suitable_model):
            raise Exception('''model.txt file doesn't exist, you can run "find_best_model" method to get it''')
        with open(self.path_to_most_suitable_model, "r") as f:
            closest_model = f.read().strip()
        self.opt_model.add_attr(closest_model, 'name')
        self.opt_model.add_attr(self.models.spec_data[closest_model]['hp_search_space'], 'hp_space')
        self.opt_model.add_attr(self.models.spec_data[closest_model]['training_configs'], 'training_configs')

    def run_inference(self):
        if not hasattr(self.opt_model, 'name'):
            logger.info("no 'update_optimal_model' method, checking for model.txt file . . . ")
            self.update_optimal_model()

        gun = Launcher(self.opt_model, remote=self.remote)
        path_to_first_checkpoint = self.path_to_best_checkpoint.split('.')[0] + str(0) + '.pt'
        gun.predict(path_to_first_checkpoint)

    def one_time_inference(self, image_path, checkpoint_path):
        from ObjectDetNet.retinanet import AdapterModel
        model = AdapterModel()
        return model.predict_single_image(checkpoint_path, image_path)


def maybe_login(env):
    try:
        dl.setenv(env)
    except:
        dl.login()
        dl.setenv(env)


def maybe_do_deployment_stuff():
    if args.deploy:
        logger.info('about to launch 2 deployments, zazu and trial')
        with open('global_configs.json', 'r') as fp:
            global_project_name = json.load(fp)['project']

        global_project = dl.projects.get(project_name=global_project_name)
        global_package_obj = push_package(global_project)
        try:
            # predict_service = deploy_predict(package=global_package_obj)
            trial_service = deploy_model(package=global_package_obj)
            zazu_service = deploy_zazu(package=global_package_obj)
            logger.info('deployments launched successfully')
        except:
            # predict_service.delete()
            trial_service.delete()
            zazu_service.delete()

    if args.zazu_timer:
        logger.info('about to launch timer deployment')
        with open('global_configs.json', 'r') as fp:
            global_project_name = json.load(fp)['project']

        global_project = dl.projects.get(project_name=global_project_name)
        global_package_obj = push_package(global_project)

        with open('configs.json', 'r') as fp:
            configs = json.load(fp)

        configs_input = dl.FunctionIO(type='Json', name='configs', value=json.dumps(configs))
        time_input = dl.FunctionIO(type='Json', name='time', value=3600*0.25)
        test_dataset_input = dl.FunctionIO(type='Json', name='test_dataset_id', value='5eb7e0bdd4eb9434c77d80b5')
        query_input = dl.FunctionIO(type='Json', name='query', value=json.dumps({"resource": "items", "sort": {}, "page": 0, "pageSize": 1000, "filter": {"$and": [{"dir": "/items/val*"}, {"hidden": False}, {"type": "file"}]}}))
        init_inputs = [configs_input, time_input, test_dataset_input, query_input]
        deploy_zazu_timer(package=global_package_obj,
                          init_inputs=init_inputs)
        logger.info('timer deployment launched successfully')

    if args.update:
        with open('global_configs.json', 'r') as fp:
            global_project_name = json.load(fp)
        maybe_login()
        global_project = dl.projects.get(project_name=global_project_name)
        update_service(global_project, 'trial')
        update_service(global_project, 'zazu')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action='store_true', default=False)
    parser.add_argument("--deploy", action='store_true', default=False)
    parser.add_argument("--update", action='store_true', default=False)
    parser.add_argument("--search", action='store_true', default=False)
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--predict", action='store_true', default=False)
    parser.add_argument("--predict_once", action='store_true', default=False)
    parser.add_argument("--zazu_timer", action='store_true', default=False)
    args = parser.parse_args()

    with open('configs.json', 'r') as fp:
        configs = json.load(fp)
    try:
        maybe_login(configs['dataloop']['setenv'])
    except:
        pass
    maybe_do_deployment_stuff()

    if args.remote:

        configs_input = dl.FunctionIO(type='Json', name='configs', value=configs)
        inputs = [configs_input]
        zazu_service = dl.services.get('zazu')
        # get project id for billing bla bla bla
        dataset_obj = get_dataset_obj(configs['dataloop'])
        id = dataset_obj.project.id

        if args.search:
            zazu_service.execute(function_name='search', execution_input=inputs, project_id=id)
        if args.predict:
            zazu_service.execute(function_name='predict', execution_input=inputs, project_id=id)

    else:
        logger = init_logging(__name__)
        this_path = path = os.getcwd()
        configs_path = os.path.join(this_path, 'configs.json')
        configs = ConfigSpec(configs_path)
        opt_model = OptModel()
        opt_model.add_child_spec(configs, 'configs')
        zazu = ZaZu(opt_model, remote=args.remote)
        if args.search:
            zazu.find_best_model()
            zazu.hp_search()
        if args.train:
            zazu.train_new_model()
        if args.predict:
            zazu.run_inference()
        if args.predict_once:
            zazu.one_time_inference('/home/noam/0120122798.jpg', 'checkpoint0.pt')
