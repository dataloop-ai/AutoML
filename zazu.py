from trial_launchpad import Launcher
from hyperparameter_tuner import Tuner, OngoingTrials
from spec import ConfigSpec, OptModel
from augmentations_tuner.fastautoaugment import augsearch
import argparse
import os
import torch
import json
from hyperparameter_tuner.trial import generate_trial_id
from logging_utils import init_logging, logginger
from object_detecter.adapter import AdapterModel
from predictor import predict
logger = logginger(__name__)


class ZaZu:
    def __init__(self, opt_model, remote=False):
        self.remote = remote
        self.opt_model = opt_model
        self.path_to_best_trial = 'best_trial.json'
        self.path_to_best_checkpoint = 'checkpoint.pt'

    def hp_search(self):
        if not self.remote:
            if self.opt_model.max_instances_at_once > torch.cuda.device_count():
                print(torch.cuda.is_available())
                raise Exception(
                    ''' 'max_instances_at_once' must be smaller or equal to the number of available gpus' ''')

        # initialize hyperparameter_tuner and gun i.e.
        ongoing_trials = OngoingTrials()
        tuner = Tuner(self.opt_model, ongoing_trials)
        gun = Launcher(self.opt_model, ongoing_trials)
        logger.info('commencing hyper-parameter search . . . ')
        tuner.search_hp()
        gun.launch_trials()
        tuner.end_trial()
        # starting second set of trials
        tuner.search_hp()
        while ongoing_trials.status is not 'STOPPED':
            gun.launch_trials()
            tuner.end_trial()
            # starting next set of trials
            tuner.search_hp()

        trials = tuner.trials
        if self.opt_model.augmentation_search_method == 'fastautoaugment':
            sorted_trial_ids = tuner.get_sorted_trial_ids()

            string1 = self.path_to_best_checkpoint.split('.')[0]
            paths_ls = []
            for i in range(len(sorted_trial_ids[:5])):
                save_checkpoint_location = string1 + str(i) + '.pt'
                logger.info(
                    'trial ' + sorted_trial_ids[i] + '\tval: ' + str(trials[sorted_trial_ids[i]]['metrics']))
                save_checkpoint_location = os.path.join(os.getcwd(), 'augmentations_tuner', 'fastautoaugment',
                                                        'FastAutoAugment', 'models', save_checkpoint_location)
                if os.path.exists(save_checkpoint_location):
                    logger.info('overwriting checkpoint . . .')
                    os.remove(save_checkpoint_location)
                torch.save(trials[sorted_trial_ids[i]]
                           ['checkpoint'], save_checkpoint_location)
                paths_ls.append(save_checkpoint_location)
            # TODO: calibrate between the model dictionaries
            aug_policy = augsearch(paths_ls=paths_ls)
            best_trial = trials[sorted_trial_ids[0]]['hp_values']
            best_trial.update({"augment_policy": aug_policy})
            metrics_and_checkpoint_dict = gun.launch_trial(
                hp_values=best_trial)
            # no oracle to create trial with, must generate on our own
            trial_id = generate_trial_id()
            tuner.add_trial(trial_id=trial_id,
                            hp_values=best_trial,
                            metrics=metrics_and_checkpoint_dict['metrics'],
                            meta_checkpoint=metrics_and_checkpoint_dict['meta_checkpoint'])

        sorted_trial_ids = tuner.get_sorted_trial_ids()
        save_best_checkpoint_location = 'best_checkpoint.pt'
        logger.info(
            'the best trial, trial ' + sorted_trial_ids[0] + '\tval: ' + str(trials[sorted_trial_ids[0]]['metrics']))
        temp_checkpoint = torch.load(
            trials[sorted_trial_ids[0]]['meta_checkpoint']['checkpoint_path'])
        checkpoint = trials[sorted_trial_ids[0]]['meta_checkpoint']
        checkpoint.update(temp_checkpoint)
        if os.path.exists(save_best_checkpoint_location):
            logger.info('overwriting checkpoint . . .')
            os.remove(save_best_checkpoint_location)
        torch.save(trials[sorted_trial_ids[0]]
                   ['meta_checkpoint'], save_best_checkpoint_location)

        logger.info('best trial: ' + str(trials[sorted_trial_ids[0]]['hp_values']) + '\nbest value: ' + str(
            trials[sorted_trial_ids[0]]['metrics']))

        best_trial = trials[sorted_trial_ids[0]]['hp_values']
        if os.path.exists(self.path_to_best_trial):
            logger.info('overwriting best_trial.json . . .')
            os.remove(self.path_to_best_trial)
        with open(self.path_to_best_trial, 'w') as fp:
            json.dump(best_trial, fp)
            logger.info('results saved to best_trial.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", action='store_true', default=False)
    parser.add_argument("--deploy", action='store_true', default=False)
    parser.add_argument("--update", action='store_true', default=False)
    parser.add_argument("--search", action='store_true', default=False)
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--predict", action='store_true', default=False)
    parser.add_argument("--zazu_timer", action='store_true', default=False)
    parser.add_argument("--checkpoint_path", type=str,
                        default='/home/noam/ZazuML/best_checkpoint.pt')
    parser.add_argument("--dataset_path", type=str, default='')
    parser.add_argument("--output_path", type=str, default='')
    args = parser.parse_args()

    with open('configs.json', 'r') as fp:
        configs = json.load(fp)
    logger = init_logging(__name__)
    this_path = path = os.getcwd()
    configs_path = os.path.join(this_path, 'configs.json')
    configs = ConfigSpec('configs.json')
    opt_model = OptModel('models.json')
    opt_model.add_child_spec(configs, 'configs')
    zazu = ZaZu(opt_model, remote=args.remote)
    if args.search:
        zazu.hp_search()
    if args.train:
        adapter = AdapterModel()
        adapter.load(checkpoint_path=args.checkpoint_path)
        adapter.train()
        print('model checkpoint is saved to: ', adapter.checkpoint_path)
    if args.predict:
        predict(pred_on_path=args.dataset_path, output_path=args.output_path,
                checkpoint_path=args.checkpoint_path, threshold=0.5)
