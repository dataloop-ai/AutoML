from tensorflow import keras
from model_selector import ModelSelector
from launch_pad import Launcher
from tuner import Tuner, OngoingTrials
from spec import ConfigSpec, ModelSpaceSpec, OptModel
import argparse
import os


def search(opt_model, remote=False):

    selector = ModelSelector(opt_model)
    selector.find_model_and_hp_search_space()

    # initialize tuner and gun i.e.
    ongoing_trials = OngoingTrials()
    tuner = Tuner(opt_model, ongoing_trials)
    gun = Launcher(opt_model, ongoing_trials, remote)

    tuner.search_hp()
    gun.launch_c()
    tuner.end_trial()

    while ongoing_trials.status is not 'STOPPED':
        tuner.search_hp()
        gun.launch_c()
        tuner.end_trial()

    return tuner.get_best_trial()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", type=int, default=0)
    args = parser.parse_args()
    # recipe includes configs
    this_path = path = os.getcwd()
    configs_path = os.path.join(this_path,'spec','samples','configs.json')
    configs = ConfigSpec(configs_path)
    model_space = ModelSpaceSpec()
    opt_model = OptModel()
    opt_model.add_child_spec(configs, 'configs')
    opt_model.add_child_spec(model_space, 'model_space')
    best_trial = search(opt_model, remote=args.remote)

    print('best trial: ', best_trial)