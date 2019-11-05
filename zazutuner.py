from tensorflow import keras
from model_selector import ModelSelector
from launch_pad.launcher import Launcher
from tuner import Tuner, OngoingTrials
import pandas as pd
from spec import RecipeSpec, DataSpec, ModelSpec


def main(data, optimal_model):
    selector = ModelSelector(optimal_model)
    selector.find_model_and_hp_search_space()

    #initialize tuner and gun i.e.
    ongoing_trials = OngoingTrials()
    tuner = Tuner(optimal_model, ongoing_trials)
    gun = Launcher(optimal_model, data, ongoing_trials)

    tuner.search_hp()
    gun.launch_c()
    tuner.end_trial()

    while True:
        tuner.search_hp()
        gun.launch_c()
        tuner.end_trial()
        if ongoing_trials.finished():
            break


    #trials = tuner.get_trials()
    #df = pd.DataFrame(trials)
    #temp_df = df.loc['metrics'].dropna()
    #best_trial_id = temp_df.idxmax()

    print('best trial', tuner.get_best_trial())


if __name__ == '__main__':
    (x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
    x = x.astype('float32') / 255.
    val_x = val_x.astype('float32') / 255.

    x = x[:10000]
    y = y[:10000]

    recipe = RecipeSpec('/Users/noam/zazu/spec/samples/recipe.json')
    opt_model = ModelSpec()

    data = DataSpec()
    data.fill(x, y)

    main(data=data, optimal_model=opt_model)

