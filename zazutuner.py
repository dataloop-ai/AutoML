from tensorflow import keras
from model_selector import ModelSelector
from launch_pad import Launcher
from tuner import Tuner, OngoingTrials
from spec import RecipeSpec, DataSpec, ModelSpaceSpec, OptModel
import argparse


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

    (x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
    x = x.astype('float32') / 255.
    val_x = val_x.astype('float32') / 255.

    x = x[:10000]
    y = y[:10000]

    recipe = RecipeSpec('/Users/noam/zazu/spec/samples/recipe.json')
    model_space = ModelSpaceSpec()
    data = DataSpec()
    data.fill(x, y)
    opt_model = OptModel()
    opt_model.add_child_spec(recipe, 'recipie')
    opt_model.add_child_spec(model_space, 'model_space')
    opt_model.add_attr_from_obj(data, 'items')
    opt_model.add_attr_from_obj(data, 'labels')
    best_trial = search(opt_model, remote=args.remote)

    print('best trial: ', best_trial)