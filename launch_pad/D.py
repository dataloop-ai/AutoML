from launch_pad.C import Experiment


class Launcher:
    def __init__(self, configs, model, data):
        self.configs = configs
        self.model = model
        self.data = data
        self.metrics_dic = {}

    def launch_c(self, trials):
        for trial_id, trial in trials.items():
            if 'metrics' not in trial:
                experiment = Experiment(trial, self.configs, self.model, self.data)
                metrics = experiment.run()
                self.metrics_dic[trial_id] = metrics
        return self.metrics_dic
