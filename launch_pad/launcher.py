from launch_pad.experiment_x import Experiment


class Launcher:
    def __init__(self, configs, model, data):
        self.configs = configs
        self.model = model
        self.data = data

    def launch_c(self, ongoing_trials):
        metrics_dic = {}
        for trial_id, trial in ongoing_trials.items():
            experiment = Experiment(trial['hp_values'], self.configs, self.model, self.data)
            metrics = experiment.run()
            metrics_dic[trial_id] = metrics
        return metrics_dic
