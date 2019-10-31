from tensorflow import keras
from model_selector import Spinner
from launch_pad.D import Launcher
from tuner import Tuner
import pandas as pd


def retdic(dic):
    if dic is not 0:
        return dic['val_accuracy']
    else:
        0


def main(task, data, priority):
    spinner = Spinner(task, priority)
    search_space, model, configs = spinner.find_closest_model_and_hp()
    #initialize tuner and gun i.e.
    tuner = Tuner(search_space, configs)
    gun = Launcher(configs, model, data)

    trials, status = tuner.search_hp()
    metrics = gun.launch_c(trials)
    while True:
        trials, status = tuner.search_hp(metrics)
        if status == 'STOPPED':
            break
        metrics = gun.launch_c(trials)
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
    main(task='detection', data=data, priority='high_accuracy_high_latency')
