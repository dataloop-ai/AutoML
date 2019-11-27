class OngoingTrials:

    def __init__(self):
        self.trials = {}
        self.status = None

    def update_trial_hp(self, trial_id, hp_values):
        self.trials[trial_id] = {'hp_values': hp_values}

    def update_status(self, status):
        self.status = status

    def update_metrics(self, trial_id, metrics):
        self.trials[trial_id]['metrics'] = metrics

    def remove_trial(self):
        # change status of trial object then
        dict_copy = self.trials.copy()
        for trial_id, _ in dict_copy.items():
            self.trials.pop(trial_id)

    @property
    def num_trials(self):
        return len(self.trials)