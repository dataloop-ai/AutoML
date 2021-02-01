
from torch.utils.data import DataLoader
from .dataloader import CustomDataset, collater
from .utils import *
import random
from keras.utils import to_categorical
import tensorflow as tf


class DataGenerator(object):
    def __init__(self, dir_path, annotation_format, task, number_classes=None, framework='pytorch', batch_size=32, dataset='train', shuffle=True, num_workers=4, annotation_path=None, function_transforms=None, built_in_transforms=None):
        self._dir_path = dir_path
        self._annotation_format = annotation_format
        self.number_classes = number_classes
        self._framework = framework
        self._batch_size = batch_size
        self._dataset = dataset
        self._shuffle = shuffle
        self._num_workers = num_workers
        self._annotation_path = annotation_path
        self._function_transforms = function_transforms
        self._built_in_transforms = built_in_transforms
        self._task = task

        self.dataset = CustomDataset(dir_path=self._dir_path, annot_format=self._annotation_format, num_categories=self.number_classes, dataset=self._dataset,
                                     annotation_path=self._annotation_path, do_task=self._task, framework_version=self._framework, function_transforms=self._function_transforms, built_in_transforms=self._built_in_transforms)

        self.loader = DataLoader(dataset=self.dataset, batch_size=self._batch_size,
                                 shuffle=self._shuffle, num_workers=self._num_workers, collate_fn=collater)

        self.iterator = iter(self.loader)
        num_data = len(self.dataset)
        self.num_batchs = num_data//batch_size
        self.count = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self.iterator

    def __next__(self):

        if self.count < self.num_batchs:
            data = next(self.iterator)
            if self._framework == 'pytorch':

                if data.task == 'classification':
                    image = data.image.permute(0, 3, 2, 1)
                    image = np.array(image)
                    label = to_categorical(
                        data.label, num_classes=self.dataset.num_classes)
                    label = np.array(label)
                    return [image, label]
                elif data.task == 'segmentation':
                    image = data.image.permute(0, 3, 2, 1)
                    image = np.array(image)
                    annotation = np.array(data.annotation)
                    mask = np.array(list(data.masks_and_category))
                    return (image, annotation, mask)
                else:
                    image = data[0]
                    target = data[1]
                    return (image, target)

            elif self._framework == 'keras':

                image = data[0]

                if self._annotation_format == 'csv' or self._annotation_format == 'txt':
                    label = data[1]
                    label = np.array(label)

                    return image, label
                elif data._object_detection:
                    annotation = np.array(data.annotation)
                    mask = np.array(list(data.masks_and_category))
                    return (image, annotation, mask)
                else:
                    annotation = np.array(data.annotation)
                    return (image, annotation)

            self.count += 1
