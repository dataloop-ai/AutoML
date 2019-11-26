import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from . import model

from retinanet.dataloaders.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, \
    Augmenter, Normalizer
from torch.utils.data import DataLoader

from . import csv_eval

print('CUDA available: {}'.format(torch.cuda.is_available()))


class RetinaModel:

    def preprocess(self, dataset='csv', csv_train=None, csv_val=None, csv_classes=None, coco_path=None,
                   resize=608):
        self.dataset = dataset
        if self.dataset == 'coco':
            if coco_path is None:
                raise ValueError('Must provide --coco_path when training on COCO,')
            self.dataset_train = CocoDataset(coco_path, set_name='train2017',
                                             transform=transforms.Compose([Normalizer(), Augmenter(), Resizer(min_side=resize)]))
            self.dataset_val = CocoDataset(coco_path, set_name='val2017',
                                           transform=transforms.Compose([Normalizer(), Resizer()]))

        elif self.dataset == 'csv':
            if csv_train is None:
                raise ValueError('Must provide --csv_train when training on COCO,')
            if csv_classes is None:
                raise ValueError('Must provide --csv_classes when training on COCO,')
            self.dataset_train = CSVDataset(train_file=csv_train, class_list=csv_classes,
                                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer(min_side=resize)])
                                            )

            if csv_val is None:
                self.dataset_val = None
                print('No validation annotations provided.')
            else:
                self.dataset_val = CSVDataset(train_file=csv_val, class_list=csv_classes,
                                              transform=transforms.Compose([Normalizer(), Resizer()]))
        else:
            raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

        sampler = AspectRatioBasedSampler(self.dataset_train, batch_size=2, drop_last=False)
        self.dataloader_train = DataLoader(self.dataset_train, num_workers=0, collate_fn=collater,
                                           batch_sampler=sampler)
        if self.dataset_val is not None:
            sampler_val = AspectRatioBasedSampler(self.dataset_val, batch_size=1, drop_last=False)
            self.dataloader_val = DataLoader(self.dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

        print('Num training images: {}'.format(len(self.dataset_train)))

    def build(self, depth=50, learning_rate=1e-5):
        # Create the model
        if depth == 18:
            retinanet = model.resnet18(num_classes=self.dataset_train.num_classes(), pretrained=True)
        elif depth == 34:
            retinanet = model.resnet34(num_classes=self.dataset_train.num_classes(), pretrained=True)
        elif depth == 50:
            retinanet = model.resnet50(num_classes=self.dataset_train.num_classes(), pretrained=True)
        elif depth == 101:
            retinanet = model.resnet101(num_classes=self.dataset_train.num_classes(), pretrained=True)
        elif depth == 152:
            retinanet = model.resnet152(num_classes=self.dataset_train.num_classes(), pretrained=True)
        else:
            raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

        self.retinanet = retinanet.cuda()
        self.retinanet.training = True
        self.optimizer = optim.Adam(self.retinanet.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

    def train(self, epochs=1):
        for epoch_num in range(epochs):

            self.retinanet.train()
            self.retinanet.freeze_bn()

            epoch_loss = []
            loss_hist = collections.deque(maxlen=500)

            for iter_num, data in enumerate(self.dataloader_train):
                try:
                    self.optimizer.zero_grad()
                    classification_loss, regression_loss = self.retinanet([data['img'].cuda().float(), data['annot'].cuda()])
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
                    print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                            epoch_num, iter_num, float(classification_loss), float(regression_loss),
                            np.mean(loss_hist)))
                    del classification_loss
                    del regression_loss
                except Exception as e:
                    print(e)
                    continue

    def get_metrics(self):
        mAP = csv_eval.evaluate(self.dataset_val, self.retinanet)
        return mAP

    def save(self):
        torch.save(self.retinanet, 'model_final.pt')


if __name__ == '__main__':
    pass
