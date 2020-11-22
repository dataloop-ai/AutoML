import sys
import os
from trial_launchpad import Launcher
from hyperparameter_tuner import Tuner, OngoingTrials
from spec import OptModel
from augmentations_tuner.fastautoaugment import augsearch
import argparse
import torch
import json
from hyperparameter_tuner.trial import generate_trial_id
from logging_utils import init_logging, logginger
from objectdetection.trial_adapter import TrialAdapter
from predictor import predict

logger = logginger(__name__)


class ZaZu(OptModel):
    def __init__(self, model_name, home_path="../data/tiny_coco", annotation_type="coco"):
        self.path_to_best_trial = 'best_trial.json'
        self.path_to_best_checkpoint = 'checkpoint.pt'
        self.name = model_name
        self.home_path = home_path
        self.annotation_type = annotation_type
        super().__init__('models.json')

    def search(self, search_method='random', epochs=2, max_trials=1,
               max_instances_at_once=1, augmentation_search=False):
        """
        max_trials: maximum number of trials before hard stop, is not used in hyperband algorithm
        """
        ongoing_trials = OngoingTrials()
        tuner = Tuner(ongoing_trials, search_method=search_method, epochs=epochs, max_trials=max_trials,
                      max_instances_at_once=max_instances_at_once, hp_space=self.hp_space)
        gun = Launcher(ongoing_trials, model_fn=self.name, training_configs=self.training_configs,
                       home_path=self.home_path, annotation_type=self.annotation_type)

        logger.info('commencing hyper-parameter search . . . ')
        tuner.search_hp()
        gun.launch_trials()
        tuner.end_trial()
        # starting second set of trials
        tuner.search_hp()
        while ongoing_trials.status != 'STOPPED':
            gun.launch_trials()
            tuner.end_trial()
            # starting next set of trials
            tuner.search_hp()

        trials = tuner.trials
        if augmentation_search:
            self._searchaugs_retrain_push(trials, tuner, gun)

        sorted_trial_ids = tuner.get_sorted_trial_ids()
        save_best_checkpoint_location = 'best_checkpoint.pt'
        logger.info(
            'the best trial, trial ' + sorted_trial_ids[0] + '\tval: ' + str(trials[sorted_trial_ids[0]]['metrics']))
        temp_checkpoint = torch.load(trials[sorted_trial_ids[0]]['meta_checkpoint']['checkpoint_path'])
        checkpoint = trials[sorted_trial_ids[0]]['meta_checkpoint']
        checkpoint.update(temp_checkpoint)
        if os.path.exists(save_best_checkpoint_location):
            logger.info('overwriting checkpoint . . .')
            os.remove(save_best_checkpoint_location)
        torch.save(trials[sorted_trial_ids[0]]['meta_checkpoint'], save_best_checkpoint_location)

        logger.info('best trial: ' + str(trials[sorted_trial_ids[0]]['hp_values']) + '\nbest value: ' + str(
            trials[sorted_trial_ids[0]]['metrics']))

        best_trial = trials[sorted_trial_ids[0]]['hp_values']
        if os.path.exists(self.path_to_best_trial):
            logger.info('overwriting best_trial.json . . .')
            os.remove(self.path_to_best_trial)
        with open(self.path_to_best_trial, 'w') as fp:
            json.dump(best_trial, fp)
            logger.info('results saved to best_trial.json')

    def _searchaugs_retrain_push(self, trials, tuner, gun):
        # search augs, retrain and upload
        sorted_trial_ids = tuner.get_sorted_trial_ids()

        string1 = self.path_to_best_checkpoint.split('.')[0]
        paths_ls = []
        for i in range(len(sorted_trial_ids[:5])):
            save_checkpoint_location = string1 + str(i) + '.pt'
            logger.info('trial ' + sorted_trial_ids[i] + '\tval: ' + str(self.trials[sorted_trial_ids[i]]['metrics']))
            save_checkpoint_location = os.path.join(os.getcwd(), 'augmentations_tuner', 'fastautoaugment',
                                                    'FastAutoAugment', 'models', save_checkpoint_location)
            if os.path.exists(save_checkpoint_location):
                logger.info('overwriting checkpoint . . .')
                os.remove(save_checkpoint_location)
            torch.save(trials[sorted_trial_ids[i]]['checkpoint'], save_checkpoint_location)
            paths_ls.append(save_checkpoint_location)
        aug_policy = augsearch(paths_ls=paths_ls)  # TODO: calibrate between the model dictionaries
        best_trial = trials[sorted_trial_ids[0]]['hp_values']
        best_trial.update({"augment_policy": aug_policy})
        metrics_and_checkpoint_dict = gun.launch_trial(hp_values=best_trial)
        # no oracle to create trial with, must generate on our own
        trial_id = generate_trial_id()
        tuner.add_trial(trial_id=trial_id,
                        hp_values=best_trial,
                        metrics=metrics_and_checkpoint_dict['metrics'],
                        meta_checkpoint=metrics_and_checkpoint_dict['meta_checkpoint'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-file", action='store_true', default=False)
    parser.add_argument("--remote", action='store_true', default=False)
    parser.add_argument("--deploy", action='store_true', default=False)
    parser.add_argument("--update", action='store_true', default=False)
    parser.add_argument("--search", action='store_true', default=False)
    parser.add_argument("--train", action='store_true', default=False)
    parser.add_argument("--predict", action='store_true', default=False)
    parser.add_argument("--zazu_timer", action='store_true', default=False)
    parser.add_argument("--checkpoint_path", type=str, default='/home/noam/ZazuML/best_checkpoint.pt')
    parser.add_argument("--dataset_path", type=str, default='')
    parser.add_argument("--output_path", type=str, default='')
    args = parser.parse_args()


    with open('configs.json', 'r') as fp:
        configs = json.load(fp)
    logger = init_logging(__name__)

    zazu = ZaZu(configs['model_name'], configs['home_path'], configs['annotation_type'])
    if args.search:
        zazu.search(configs['search_method'], configs['epochs'], configs['max_trials'],
                    configs['max_instances_at_once'], configs['augmentation_search'])
    if args.train:
        adapter = TrialAdapter(0)
        adapter.load(checkpoint_path=args.checkpoint_path)
        adapter.train()
        print('model checkpoint is saved to: ', adapter.checkpoint_path)
    if args.predict:
        predict(pred_on_path=args.dataset_path, output_path=args.output_path,
                checkpoint_path=args.checkpoint_path, threshold=0.5)
