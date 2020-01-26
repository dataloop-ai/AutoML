import math
from .oracle import Oracle

class HyperBand(Oracle):

    def __init__(self,
                 space,
                 max_epochs,
                 factor=3):
        super().__init__(space)
        self.max_epochs = max_epochs
        # Minimum epochs before successive halving, Hyperband sweeps through varying
        # degress of aggressiveness.
        self.min_epochs = 1
        self.factor = factor
        self._max_collisions = 20
        self._current_iteration = 0
        self.hyperband_iterations = 1
        self.s_max = int(math.log(max_epochs, factor))
        # Start with most aggressively halving bracket.
        self._current_bracket_num = self.s_max
        self._start_new_bracket()

    def _populate_space(self, trial_id):
        self._reset_bracket_if_finished()

        if self._bracket:
            values = self._get_trial(trial_id)
            return {'status': 'RUNNING', 'values': values}
        # This is reached if no trials from current brackets can be run.

        # Max sweeps has been reached, no more brackets should be created.
        elif (self._current_bracket_num == 0 and
                self._current_iteration + 1 == self.hyperband_iterations):
            self._increment_bracket_num()
            return {'status': 'STOPPED'}
        # Create a new bracket.
        else:
            self._increment_bracket_num()
            self._start_new_bracket()
            values = self._get_trial(trial_id)
            return {'status': 'RUNNING', 'values': values}

    def _get_trial(self, trial_id):
        bracket_num = self._bracket['bracket_num']
        rounds = self._bracket['rounds']
        # are we still filling the first bracket ?
        if len(rounds[0]) < self._get_size(bracket_num, round_num=0):
            # Populate the initial random trials for this bracket.
            return self._random_trial(trial_id, self._bracket)
        else:
            # Try to populate incomplete rounds for this bracket.
            for round_num in range(1, len(rounds)):
                round_info = rounds[round_num]
                past_round_info = rounds[round_num - 1]
                size = self._get_size(bracket_num, round_num)
                past_size = self._get_size(bracket_num, round_num - 1)

                # If more trials from the last round are ready than will be
                # thrown out, we can select the best to run for the next round.
                already_selected = [info['past_id'] for info in round_info]
                candidates = [self.trials[info['id']]
                              for info in past_round_info
                              if info['id'] not in already_selected]
                candidates = [t for t in candidates if t.status == 'COMPLETED']
                if len(candidates) > past_size - size:
                    sorted_candidates = sorted(
                        candidates,
                        key=lambda t: t.score,
                        reverse=self.objective.direction == 'max')
                    best_trial = sorted_candidates[0]

                    values = best_trial.hyperparameters.values.copy()
                    values['tuner/trial_id'] = best_trial.trial_id
                    values['tuner/epochs'] = self._get_epochs(
                        bracket_num, round_num)
                    values['tuner/initial_epoch'] = self._get_epochs(
                        bracket_num, round_num - 1)
                    values['tuner/bracket'] = self._current_bracket_num
                    values['tuner/round'] = round_num

                    round_info.append({'past_id': best_trial.trial_id,
                                       'id': trial_id})
                    return values


    def _start_new_bracket(self):
        rounds = []
        for _ in range(self._current_bracket_num + 1):
            rounds.append([])
        bracket = {'bracket_num': self._current_bracket_num, 'rounds': rounds}
        self._bracket = bracket

    def _reset_bracket_if_finished(self):
        bracket_num = self._bracket['bracket_num']
        rounds = self._bracket['rounds']
        last_round = len(rounds) - 1
        if len(rounds[last_round]) == self._get_size(bracket_num, last_round):
            # All trials have been created for the current bracket. so reset it.
            self._bracket = {}
        return self._bracket

    def _get_size(self, bracket_num, round_num):
        # Set up so that each bracket takes approx. the same amount of resources.
        bracket0_to_bracketend_ratio = (self.s_max + 1) / (bracket_num + 1)
        return math.ceil(bracket0_to_bracketend_ratio * self.factor**(bracket_num - round_num))

    def _increment_bracket_num(self):
        self._current_bracket_num -= 1
        if self._current_bracket_num < 0:
            self._current_bracket_num = self.s_max
            self._current_iteration += 1

    def _random_trial(self, trial_id, bracket):
        bracket_num = bracket['bracket_num']
        rounds = bracket['rounds']
        values = super()._populate_space(trial_id)['values']
        if values:
            values['tuner/epochs'] = self._get_epochs(bracket_num, 0)
            values['tuner/initial_epoch'] = 0
            values['tuner/bracket'] = self._current_bracket
            values['tuner/round'] = 0
            rounds[0].append({'past_id': None, 'id': trial_id})
            return {'status': 'RUNNING', 'values': values}
        elif self.ongoing_trials:
            # Can't create new random values, but successive halvings may still
            # be needed.
            return {'status': 'IDLE'}
        else:
            # Collision and no ongoing trials should trigger an exit.
            return {'status': 'STOPPED'}

    def _get_epochs(self, bracket_num, round_num):
        return math.ceil(self.max_epochs / self.factor**(bracket_num - round_num))