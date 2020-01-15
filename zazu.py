from model_selector import find_model
from launch_pad import Launcher
from tuner import Tuner, OngoingTrials
from spec import ConfigSpec, OptModel
from spec import ModelsSpec
from plugin_utils import maybe_download_data, get_dataset_obj
import argparse
import os
import torch
import json
import logging

logger = logging.getLogger('Zazu')


class ZaZu:
    def __init__(self, opt_model, remote=False):
        self.remote = remote
        self.opt_model = opt_model
        self.path_to_most_suitable_model = 'model.txt'
        self.path_to_best_trial = 'best_trial.json'
        self.path_to_best_checkpoint = 'checkpoint.pt'
        models_spec_path = 'models.json'
        self.models = ModelsSpec(models_spec_path)

    def find_best_model(self):
        closest_model = find_model(self.opt_model, self.models)
        logger.info('closest model to your preferences is: ', closest_model)

        if os.path.exists(self.path_to_most_suitable_model):
            logger.info('overwriting model.txt . . .')
            os.remove(self.path_to_most_suitable_model)
        with open(self.path_to_most_suitable_model, "w") as f:
            f.write(closest_model)
        self.update_optimal_model()

    def hp_search(self):
        if not self.remote:
            if self.opt_model.max_instances_at_once > torch.cuda.device_count():
                raise Exception(''' 'max_instances_at_once' must be smaller or equal to the number of available gpus''')
        if not hasattr(self.opt_model, 'name'):
            logger.info("no 'update_optimal_model' method, checking for model.txt file . . . ")
            self.update_optimal_model()
        # initialize tuner and gun i.e.
        ongoing_trials = OngoingTrials()
        tuner = Tuner(self.opt_model, ongoing_trials)
        gun = Launcher(self.opt_model, ongoing_trials, remote=self.remote)
        logger.info('commencing hyper-parameter search . . . ')
        tuner.search_hp()
        gun.launch_trials()
        tuner.end_trial()

        while ongoing_trials.status is not 'STOPPED':
            tuner.search_hp()
            gun.launch_trials()
            tuner.end_trial()

        best_trial = tuner.get_best_trial()
        logger.info('best trial: ', best_trial)
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

        gun = Launcher(self.opt_model)
        gun.predict(self.path_to_best_checkpoint)


def dataloop_login(token_path):
    import dtlpy as dl
    if not os.path.exists(token_path):
        raise Exception('''must have a token in ''' + token_path)
    with open(token_path, "r") as f:
        token = f.read().strip()
    try:
        dl.login_token(token)
    except Exception as e:
        new_token = input("token timed out, enter new token: ")
        os.remove(token_path)
        with open(token_path, "w") as f:
            f.write(new_token)
    dl.setenv('dev')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action='store_true', default=False)
    parser.add_argument("--search", action='store_true', default=False)
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--predict", action='store_true', default=False)
    args = parser.parse_args()
    if args.remote:
        dataloop_login(token_path='token.txt')
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
