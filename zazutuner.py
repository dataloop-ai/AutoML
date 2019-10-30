from tensorflow import keras
from A import Spinner
from B import Tuner
from D import Launcher
import pandas as pd


def retdic(dic):
    if dic is not 0:
        return dic['val_accuracy']
    else:
        0


def main(task, data, priority):
    spinner = Spinner(task, priority)
    search_space, model, configs = spinner.find_closest_model_and_hp()
    #initialize tuner and gun i.e. B and D
    tuner = Tuner(search_space, configs)
    gun = Launcher(configs, model, data)

    trials, status = tuner.search_hp()
    metrics = gun.launch_c(trials)
    while True:
        if status == 'STOPPED':
            break
        trials, status = tuner.search_hp(metrics)
        metrics = gun.launch_c(trials)
    df = pd.DataFrame(trials)
    temp_df = df.loc['metrics'].fillna(0).apply(retdic)
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
