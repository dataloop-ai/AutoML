from .oracle import Oracle
from .hyperband import HyperBand
import pandas as pd


class Tuner:

    def __init__(self, optimal_model, ongoing_trials):

        if optimal_model.search_method == "hyperband":
            self.oracle = HyperBand(space=optimal_model.hp_space, max_epochs=optimal_model.epochs)
        elif optimal_model.search_method == "random_search":
            self.oracle = Oracle(space=optimal_model.hp_space, max_trials=optimal_model.max_trials)
        else:
            raise Exception('have not defined proper search_method param in configs.json')

        self.ongoing_trials = ongoing_trials
        self.max_instances_at_once = optimal_model.max_instances_at_once

    def end_trial(self):
        self.oracle.update_metrics(self.ongoing_trials.trials)
        self.ongoing_trials.remove_trial()

    def search_hp(self):

        for _ in range(self.max_instances_at_once):
            trial_id, hp_values, status = self.oracle.create_trial()
            self.ongoing_trials.update_status(status)
            if status == 'STOPPED':
                break
            self.ongoing_trials.update_trial_hp(trial_id, hp_values=hp_values)

    def get_trials(self):
        return self.oracle.trials

    def get_best_trial(self):
        df = pd.DataFrame(self.oracle.trials)
        temp_df = df.loc['metrics'].dropna()
        best_trial_id = temp_df.apply(lambda x: x['val_accuracy']).idxmax()
        return self.oracle.trials[best_trial_id]