from .trial import generate_trial_id
import random
import hashlib
import pandas as pd


class Oracle:

    def __init__(self, space, max_epochs, max_trials=None):
        self.max_epochs = max_epochs
        self.space = space
        self.trials = {}
        self._tried_so_far = set()
        self.max_trials = max_trials
        self.are_metrics = False
        self._max_collisions = 20

    def create_trial(self):
        trial_id = generate_trial_id()
        if self.are_metrics:
            df = pd.DataFrame(self.trials)
            temp_df = df.loc['metrics'].dropna().apply(lambda x: x['val_accuracy'])
        if self.max_trials is not None and len(self.trials) >= self.max_trials:
            status = 'STOPPED'
            values = None
        elif self.are_metrics and temp_df.max() > 0.998:
            status = 'STOPPED'
            values = None
        else:
            response = self._populate_space(trial_id)
            if response is None:
                status = 'STOPPED'
                values = None
            else:
                status = response['status']
                values = response['values'] if 'values' in response else None
                if values is not None:
                    self.trials[trial_id] = {'hp_values': values}

        return trial_id, values, status

    def update_metrics(self, ongoing_trials):
        for trial_id, trial in ongoing_trials.items():
            self.trials[trial_id]['metrics'] = trial['metrics']
            self.trials[trial_id]['meta_checkpoint'] = trial['meta_checkpoint']

    def _populate_space(self, trial_id):

        collisions = 0
        while 1:
            # Generate a set of random values.
            values = {}
            for p in self.space:
                values[p['name']] = random.choice(p['values'])

            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                collisions += 1
                if collisions > self._max_collisions:
                    return None
                continue
            self._tried_so_far.add(values_hash)
            values['hyperparameter_tuner/new_trial_id'] = trial_id
            values['hyperparameter_tuner/epochs'] = self.max_epochs
            values['hyperparameter_tuner/initial_epoch'] = 0
            break
        return {'status': 'RUNNING',
                'values': values}

    def _compute_values_hash(self, values):
        keys = sorted(values.keys())
        s = ''.join(str(k) + '=' + str(values[k]) for k in keys)
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]
