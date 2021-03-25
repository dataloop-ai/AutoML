
from torch.utils.data import DataLoader
from .dataloader import CustomDataset, collater
from .utils import *
import random
from keras.utils import to_categorical
import tensorflow as tf


class DataGenerator(object):
    """
    dir_path: the path of the dataset directory
    annotation_format: the annotataion file format can be 'yolo' or coco'
    task: input 'classification' or 'detection' or 'segmentation'
    framework: use 'pytorch' or 'keras' version 
    batch_size: a number that a batch can contain the data each time
    dataset: mark as train data or test data
    shuffle: suffle the data or not
    num_workers: how many workers for multiprocessing
    annotation_path: point to the path of annotation file directly
    function_transforms: a string/list of transform function(s)
    built_in_transforms: a string/list of built intrain
    num_classes: number of categories
    labels: list all the classes name 


    """

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
        self.loader = []

        self.dataset = CustomDataset(dir_path=self._dir_path, annot_format=self._annotation_format, num_categories=self.number_classes, dataset=self._dataset,
                                     annotation_path=self._annotation_path, do_task=self._task, framework_version=self._framework, function_transforms=self._function_transforms, built_in_transforms=self._built_in_transforms)

        self.loader = DataLoader(dataset=self.dataset, batch_size=self._batch_size,
                                 shuffle=self._shuffle, num_workers=self._num_workers, collate_fn=collater)

        self.num_classes = self.dataset.num_classes

        if self._annotation_format == 'coco':
            self.labels = self.dataset.labels.values()
        else:
            self.labels = self.dataset.labels

        self.iterator = iter(self.loader)
        num_data = len(self.dataset)
        self.num_batchs = num_data//batch_size
        self.count = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return dataGeneratorIterator(self)


class dataGeneratorIterator():
    def __init__(self, dataGenerator):
        self.count = dataGenerator.count
        self.num_batchs = dataGenerator.num_batchs
        self._framework = dataGenerator._framework
        self._task = dataGenerator._task
        self.dataset = dataGenerator.dataset
        self.iterator = dataGenerator.iterator

    def __next__(self):

        if self.count < self.num_batchs:
            data = next(self.iterator)
            if self._framework == 'pytorch':
                # image = data.image.permute(0, 3, 2, 1)
                # image = np.array(image)
                image = data.image
                if self._task == 'classification':

                    label = data.label

                    return image, label
                elif self._task == 'segmentation':
                    annotation = np.array(data.annotation)
                    mask = np.array(list(data.masks_and_category))
                    return image, annotation, mask
                elif self._task == 'detection':
                    target = data.target
                    return image, target

            elif self._framework == 'keras':

                if self._task == 'classification':
                    image = data[0]
                    label = data[1]
                    label = np.array(label)

                    return image, label
                elif self._task == 'segmentation':
                    image = data.image
                    annotation = np.array(data.annotation)
                    mask = np.array(list(data.masks_and_category))
                    return (image, annotation, mask)
                elif self._task == 'detection':
                    image = data[0]
                    target = data[1]
                    return (image, target)

            self.count += 1
