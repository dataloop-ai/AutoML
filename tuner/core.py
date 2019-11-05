from .oracle import Oracle
from .ongoingtrials import OngoingTrials

class Tuner:

    def __init__(self, optimal_model, ongoing_trials):

        self.oracle = Oracle(space=optimal_model.hp_space, configs=optimal_model.configs)
        self.ongoing_trials = ongoing_trials
        self.max_instances_at_once = optimal_model.configs['max_instances_at_once']

    def end_trial(self):
        self.oracle.update_metrics(self.ongoing_trials.trials)
        self.ongoing_trials.remove_trial()

    def search_hp(self):

        for _ in range(self.max_instances_at_once):
            trial_id, hp_values, status = self.oracle.create_trial()
            if status == 'STOPPED':
                break
            else:

                self.ongoing_trials.update_trial_hp(trial_id, hp_values=hp_values)
        return self.ongoing_trials, status

    def get_trials(self):
        return self.oracle.trials
