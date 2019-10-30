from oracle import Oracle


class Tuner:

    def __init__(self, search_space, configurations):

        self.oracle = Oracle(space=search_space, config=configurations)
        self.max_instances_at_once = configurations['max_instances_at_once']
    def search_hp(self, metrics=None):
        if metrics is not None:
            self.oracle.update(metrics)

        for _ in range(self.max_instances_at_once):
            trials, status = self.oracle.create_trial()
            if status == 'STOPPED':
                break

        return trials, status
