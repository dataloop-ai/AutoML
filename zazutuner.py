from tensorflow import keras
from model_selector import ModelSelector
from launch_pad.launcher import Launcher
from tuner import Tuner, Oracle
import pandas as pd



def main(recipe, data, priority):
    selector = ModelSelector(modelSelectionSpec)
    search_space, model, configs = selector.find_closest_model_and_hp()
    #initialize tuner and gun i.e.
    tuner = Tuner(search_space, configs)
    gun = Launcher(configs, model, data)

    ongoing_trials, status = tuner.search_hp()
    metrics = gun.launch_c(ongoing_trials)
    tuner.update_metrics(metrics)
    while True:
        ongoing_trials, status = tuner.search_hp()
        metrics = gun.launch_c(ongoing_trials)
        tuner.update_metrics(metrics)
        if status == 'STOPPED':
            break


    trials = tuner.get_trials()
    df = pd.DataFrame(trials)
    temp_df = df.loc['metrics'].dropna()
    best_trial_id = temp_df.idxmax()

    print('best trial', trials[best_trial_id])


if __name__ == '__main__':
    (x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
    x = x.astype('float32') / 255.
    val_x = val_x.astype('float32') / 255.

    x = x[:10000]
    y = y[:10000]

    data = {'images': x, 'labels': y}
    main(recipe='detection', data=data, priority='high_accuracy_high_latency')
