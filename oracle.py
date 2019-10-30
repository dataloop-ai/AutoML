from kerastuner.engine import trial as trial_lib
import random
import hashlib
import pandas as pd

class Oracle:

    def __init__(self, space, config):
        self.space = space
        self.trials = {}
        self.ongoing_trials = {}
        self._tried_so_far = set()
        self.max_trials = config['max_trials']
        self.metrics = None

    def create_trial(self):
        trial_id = trial_lib.generate_trial_id()
        if self.metrics is not None:
            df = pd.DataFrame(self.trials)
            temp_df = df.loc['metrics'].apply(lambda x: x['val_accuracy'])
        if len(self.trials) >= self.max_trials:
            status = 'STOPPED'
        elif self.metrics is not None and temp_df.max() > 0.998:
            status = 'STOPPED'
        else:
            response = self._populate_space(trial_id)
            status = response['status']
            values = response['values'] if 'values' in response else None

            self.trials[trial_id] = values

        return self.trials, status

    def update(self, metrics):
        for metric in metrics:
            self.trials[metric[0]]['metrics'] = metric[1]


    def _populate_space(self, _):
        while 1:
            # Generate a set of random values.
            values = {}
            for p in self.space:
                values[p['name']] = random.choice(p['values'])

            values_hash = self._compute_values_hash(values)
            if values_hash in self._tried_so_far:
                continue
            self._tried_so_far.add(values_hash)
            break
        return {'status': 'RUNNING',
                'values': values}

    def _compute_values_hash(self, values):
        keys = sorted(values.keys())
        s = ''.join(str(k) + '=' + str(values[k]) for k in keys)
        return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]