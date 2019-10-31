from tuner.oracle import Oracle


class Tuner:

    def __init__(self, search_space, configurations):

        self.oracle = Oracle(space=search_space, config=configurations)
        self.max_instances_at_once = configurations['max_instances_at_once']

    def update_metrics(self, metrics):
        self.oracle.update(metrics)

    def search_hp(self):
        ongoing_trials = {}
        for _ in range(self.max_instances_at_once):
            trial_id, hp_values, status = self.oracle.create_trial()
            if status == 'STOPPED':
                break
            else:
                ongoing_trials[trial_id] = hp_values

        return ongoing_trials, status

    def get_trials(self):
        return self.oracle.trials
