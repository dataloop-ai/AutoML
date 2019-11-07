import hashlib
import random
import time


def generate_trial_id():
    s = str(time.time()) + str(random.randint(1, 1e7))
    return hashlib.sha256(s.encode('utf-8')).hexdigest()[:32]


class Trial:

    def __init__(self, trial_id, hp_values, status):
        self.trial_id = trial_id
        self.hp_values = hp_values
        self.status = status
        self.metrics = {}

    def load_state(self, state):
        self.trial_id = state['trial_id']
        self.hp_values = state['hp_values']
        self.metrics = state['metrics']
        self.status = state['status']

    def get_state(self):
        state_dict = {
            'trial_id': self.trial_id,
            'hp_values': self.hp_values,
            'metrics': self.metrics,
            'status': self.status
        }
        return state_dict
