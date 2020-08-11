import collections
import os
import glob
import shutil
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
if __package__ == '':
    import model, csv_eval
    from dataloaders import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, \
        Augmenter, Normalizer
else:
    from . import model, csv_eval
    from .dataloaders import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, \
        Augmenter, Normalizer
from torch.utils.data import DataLoader

try:
    from logging_utils import logginger
    logger = logginger(__name__)
except:
    import logging
    logger = logging.getLogger(__name__)

print('CUDA available: {}'.format(torch.cuda.is_available()))


class RetinaModel:
    def __init__(self, device_index, home_path, save_trial_id, resume_trial_id=None, checkpoint=None):
        if os.getcwd().split('/')[-1] == 'ObjectDetNet':
            home_path = os.path.join('..', home_path)
        self.home_path = home_path
        self.device = torch.device(type='cuda', index=device_index) if torch.cuda.is_available() else torch.device(type='cpu')
        self.checkpoint = checkpoint
        if os.getcwd().split('/')[-1] == 'ObjectDetNet':
            this_path = os.path.join(os.getcwd(), 'retinanet')
        else:
            this_path = os.path.join(os.getcwd(), 'ObjectDetNet/retinanet')
        self.weights_dir_path = os.path.join(this_path, 'weights')

        if resume_trial_id:
            assert (checkpoint == None), "you can't load checkpoint and also resume given a past trial id"
            resume_last_checkpoint_path = os.path.join(this_path, 'weights', 'last_' + resume_trial_id + '.pt')
            resume_best_checkpoint_path = os.path.join(this_path, 'weights', 'best_' + resume_trial_id + '.pt')
            self.checkpoint = torch.load(resume_last_checkpoint_path)
            # TODO: resume from best???
        self.save_last_checkpoint_path = os.path.join(this_path, 'weights', 'last_' + save_trial_id + '.pt')
        self.save_best_checkpoint_path = os.path.join(this_path, 'weights', 'best_' + save_trial_id + '.pt')
        self.save_trial_id = save_trial_id
        self.results_path = os.path.join(this_path, 'weights', 'results.txt')

        self.best_fitness = - float('inf')
        self.tb_writer = None
        self.retinanet = None

    def preprocess(self, dataset='csv', csv_train=None, csv_val=None, csv_classes=None, coco_path=False,
                   train_set_name='train2017', val_set_name='val2017', resize=608):
        self.dataset = dataset
        if self.dataset == 'coco':
            if coco_path is None:
                raise ValueError('Must provide --home_path when training on COCO,')
            self.dataset_train = CocoDataset(self.home_path, set_name=train_set_name,
                                             transform=transforms.Compose(
                                                 [Normalizer(), Augmenter(), Resizer(min_side=resize)]))
            self.dataset_val = CocoDataset(self.home_path, set_name=val_set_name,
                                           transform=transforms.Compose([Normalizer(), Resizer(min_side=resize)]))

        elif self.dataset == 'csv':
            if csv_train is None:
                raise ValueError('Must provide --csv_train when training on COCO,')
            if csv_classes is None:
                raise ValueError('Must provide --csv_classes when training on COCO,')
            self.dataset_train = CSVDataset(train_file=csv_train, class_list=csv_classes,
                                            transform=transforms.Compose(
                                                [Normalizer(), Augmenter(), Resizer(min_side=resize)])
                                            )

            if csv_val is None:
                self.dataset_val = None
                print('No validation annotations provided.')
            else:
                self.dataset_val = CSVDataset(train_file=csv_val, class_list=csv_classes,
                                              transform=transforms.Compose([Normalizer(), Resizer(min_side=resize)]))
        else:
            raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

        sampler = AspectRatioBasedSampler(self.dataset_train, batch_size=2, drop_last=False)
        self.dataloader_train = DataLoader(self.dataset_train, num_workers=0, collate_fn=collater,
                                           batch_sampler=sampler)
        if self.dataset_val is not None:
            sampler_val = AspectRatioBasedSampler(self.dataset_val, batch_size=1, drop_last=False)
            self.dataloader_val = DataLoader(self.dataset_val, num_workers=3, collate_fn=collater,
                                             batch_sampler=sampler_val)

        print('Num training images: {}'.format(len(self.dataset_train)))
        if len(self.dataset_val) == 0:
            raise Exception('num val images is 0!')
        print('Num val images: {}'.format(len(self.dataset_val)))

    def build(self, depth=50, learning_rate=1e-5, ratios=[0.5, 1, 2],
              scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]):
        # Create the model
        if depth == 18:
            retinanet = model.resnet18(num_classes=self.dataset_train.num_classes(), ratios=ratios, scales=scales,
                                       weights_dir=self.weights_dir_path,
                                       pretrained=True)
        elif depth == 34:
            retinanet = model.resnet34(num_classes=self.dataset_train.num_classes(), ratios=ratios, scales=scales,
                                       weights_dir=self.weights_dir_path,
                                       pretrained=True)
        elif depth == 50:
            retinanet = model.resnet50(num_classes=self.dataset_train.num_classes(), ratios=ratios, scales=scales,
                                       weights_dir=self.weights_dir_path,
                                       pretrained=True)
        elif depth == 101:
            retinanet = model.resnet101(num_classes=self.dataset_train.num_classes(), ratios=ratios, scales=scales,
                                        weights_dir=self.weights_dir_path,
                                        pretrained=True)
        elif depth == 152:
            retinanet = model.resnet152(num_classes=self.dataset_train.num_classes(), ratios=ratios, scales=scales,
                                        weights_dir=self.weights_dir_path,
                                        pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')
        self.retinanet = retinanet.to(device=self.device)
        self.retinanet.training = True
        self.optimizer = optim.Adam(self.retinanet.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

        if self.checkpoint is not None:
            self.retinanet.load_state_dict(self.checkpoint['model'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.scheduler.load_state_dict(self.checkpoint['scheduler'])  # TODO: test this, is it done right?
            # TODO is it right to resume_read_trial optimizer and schedular like this???
        self.ratios = ratios
        self.scales = scales
        self.depth = depth

    def train(self, epochs=100, init_epoch=0):

        # Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/
        from torch.utils.tensorboard import SummaryWriter
        self.tb_writer = SummaryWriter(comment=self.save_trial_id[:3])
        for epoch_num in range(init_epoch + 1, epochs + 1):

            print('total epochs: ', epochs)
            self.retinanet.train()
            self.retinanet.freeze_bn()

            epoch_loss = []
            loss_hist = collections.deque(maxlen=500)
            total_num_iterations = len(self.dataloader_train)
            dataloader_iterator = iter(self.dataloader_train)
            pbar = tqdm(total=total_num_iterations)

            for iter_num in range(1, total_num_iterations + 1):
                try:
                    data = next(dataloader_iterator)
                    self.optimizer.zero_grad()
                    classification_loss, regression_loss = self.retinanet(
                        [data['img'].to(device=self.device).float(), data['annot'].to(device=self.device)])
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()
                    loss = classification_loss + regression_loss
                    if bool(loss == 0):
                        continue
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.retinanet.parameters(), 0.1)
                    self.optimizer.step()
                    loss_hist.append(float(loss))
                    epoch_loss.append(float(loss))
                    s = 'Trial {} -- Epoch: {}/{} | Iteration: {}/{}  | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        self.save_trial_id[:3], epoch_num, epochs, iter_num, total_num_iterations,
                        float(classification_loss),
                        float(regression_loss), np.mean(loss_hist))
                    pbar.set_description(s)
                    pbar.update()
                    del classification_loss
                    del regression_loss
                except Exception as e:
                    logger.info(e)
                    pbar.update()
                    continue
            pbar.close()
            self.scheduler.step(np.mean(epoch_loss))
            self.final_epoch = epoch_num == epochs

            mAP = csv_eval.evaluate(self.dataset_val, self.retinanet)
            self._write_to_tensorboard(mAP, np.mean(loss_hist), epoch_num)

            self._save_checkpoint(mAP, epoch_num)
            if self.final_epoch:
                self._save_classes_for_inference()

    def get_best_checkpoint(self):
        return torch.load(self.save_best_checkpoint_path)

    def get_best_metrics(self):
        checkpoint = torch.load(self.save_best_checkpoint_path)
        self.retinanet.load_state_dict(checkpoint['model'])
        mAP = csv_eval.evaluate(self.dataset_val, self.retinanet)
        return mAP.item()

    def get_best_metrics_and_checkpoint(self):
        return {'metrics': {'val_accuracy': self.get_best_metrics()},
                'checkpoint': self.get_best_checkpoint()}

    def save(self):
        torch.save(self.retinanet, 'model_final.pt')

    def _save_classes_for_inference(self):
        classes_path = os.path.join(self.home_path, "d.names")
        if os.path.exists(classes_path):
            os.remove(classes_path)
        print("saving classes to be used later for inference at ", classes_path)
        with open(classes_path, "w") as f:
            for key in self.dataset_train.classes.keys():
                f.write(key)
                f.write("\n")

    def _write_to_tensorboard(self, results, mloss, epoch):

        # Write Tensorboard results
        if self.tb_writer:
            x = [mloss.item()] + [results.item()]
            titles = ['Train_Loss', '0.5AP']
            for xi, title in zip(x, titles):
                self.tb_writer.add_scalar(title, xi, epoch)

    def _save_checkpoint(self, results, epoch):

        # Update best mAP
        fitness = results  # total loss
        if fitness > self.best_fitness:
            self.best_fitness = fitness

        # Create checkpoint
        checkpoint = {'epoch': epoch,
                      'metrics': {'val_accuracy': results.item()},
                      'model': self.retinanet.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'labels': self.dataset_train.labels
                      }

        # Save last checkpoint
        torch.save(checkpoint, self.save_last_checkpoint_path)

        # Save best checkpoint
        if self.best_fitness == fitness:
            torch.save(checkpoint, self.save_best_checkpoint_path)

        # Delete checkpoint
        del checkpoint

    def delete_stuff(self):
        files_ls = glob.glob(os.path.join(self.weights_dir_path, 'l*'))
        files_ls += glob.glob(os.path.join(self.weights_dir_path, 'b*'))
        for file in files_ls:
            try:
                os.remove(file)
            except:
                logger.info("Error while deleting file : " + file)
        shutil.rmtree(os.path.join(os.getcwd(), 'runs'))
if __name__ == '__main__':
    pass
