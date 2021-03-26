
from .compute_overlap import compute_overlap
import copy
import csv
import glob
import os
import random
import sys

import cv2
import numpy as np
import torch
from PIL.Image import Image


import skimage
import skimage.color
import skimage.io
import skimage.transform
from pycocotools import coco
from pycocotools.coco import COCO
from torch.utils.data import Dataset, Sampler

from .utils import draw_bbox
from importlib import import_module
from .image import ImageData
from .custom_transforms import *
import pandas as pd
from keras.utils import to_categorical
import tensorflow as tf
import keras
import pyximport
pyximport.install()


np.set_printoptions(suppress=True)

augmentations = ['Translate_Y',
                 'Translate_Y_BBoxes',
                 'Translate_X',
                 'Translate_X_BBoxes',
                 'CutOut',
                 'CutOut_BBoxes',
                 'Rotate',
                 'ShearX',
                 'ShearX_BBoxes',
                 'ShearY',
                 'ShearY_BBoxes',
                 'Equalize',
                 'Equalize_BBoxes',
                 'Solarize',
                 'Solarize_BBoxes',
                 'Color',
                 'Color_BBoxes',
                 'FlipLR'
                 ]


class CustomDataset(Dataset):
    def __init__(self, dir_path, annot_format, do_task=None, framework_version=None, annotation_path=None, function_transforms=None, built_in_transforms=None, dataset="train"):
        """
        Args:
            dir_path: the dataset path, if the annotation_path is None, then it will search all the files, if the annotation is not None, it will search the images.
            annot_format: annotation format. i.e. CoCo or Yolo.
            annotation_path: if has, input the .json or txt annotation path directly
            functions_of_transform" is a list of string with single/variety of transform functions.
            built_in_augmentations: is a list of string with single/variety of augmentations in the library.
            dataset is train or test
        """

        self.dir_path = dir_path
        self.annot_format = annot_format
        self.annot_path = annotation_path
        self.function_transforms = function_transforms
        self.built_in_transforms = built_in_transforms
        self.dataset = dataset
        self.img_path_list = []
        self.img_file_list = []
        self.img = []
        self.annot = []

        self.framework_version = framework_version
        self.do_task = do_task
        self.box = []
        self.label = []
        self.mask = []
        self.image_id = []
        self.area = []
        self.iscrowd = []
        self.target = {}

        self.ann_path_list = []
        if self.annot_format == 'yolo':
            # get image list
            self.img_path_list = glob.glob(dir_path + '/' + '*.jpg')+glob.glob(
                dir_path + '/' + '*.jpeg')+glob.glob(dir_path + '/' + '*.png')
            if self.annot_path is None:
                # get annotation list
                self.ann_path_list = glob.glob(dir_path + '/'+'*.txt')
            else:
                self.ann_path_list = self.annot_path

            self.classes_set = set()
            self.calculate_classes()
            self.img_path_list.sort()
            self.ann_path_list.sort()

        # get all the image file path
        elif self.annot_format == 'coco':
            self.set_name = 'train'
            self.img_path_list = glob.glob(
                dir_path + '/images/' + self.set_name + '/'+'*.jpg')

            if self.annot_path is None:
                self.coco = COCO(os.path.join(
                    self.dir_path, 'annotations', 'instances_' + self.set_name + '.json'))
            else:
                self.coco = COCO(
                    self.annot_path)
            self.image_ids = self.coco.getImgIds()
            self.load_classes()
            self.img_path_list.sort()
            self.ann_path_list.sort()

        elif self.annot_format == 'csv':
            if self.annot_path is None:
                self.dir_path_list = glob.glob(self.dir_path + '/' + '*.csv')
            else:
                self.dir_path_list = self.annot_path
            self.data = pd.read_csv(self.dir_path_list[0], header=None)
            self.data.columns = ['0', '1']
            self.load_all_pictures()
            self.load_all_classes()

        # read the txt file and separate to two culomn (filename & category)and save as dataFrame
        elif self.annot_format == 'txt':
            column_zero = []
            column_one = []
            if self.annot_path is None:
                self.dir_path_list = glob.glob(self.dir_path + '/' + '*.txt')
            else:
                self.dir_path_list = glob.glob(annotation_path)
            with open(self.dir_path_list[0]) as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    split_list = line.split(" ")
                    column_zero.append(split_list.pop(0))
                    column_one += split_list

            dict = {'0': column_zero, '1': column_one}
            self.data = pd.DataFrame(dict)
            self.load_all_pictures()
            self.load_all_classes()
    # list all the category name

    def load_all_classes(self):
        category = set(self.data['1'].to_list())
        self.labels = list(category)
    # from all folder ,get all the pictures

    def load_all_pictures(self):
        if self.dataset == 'train':

            for index in range(len(self.data['0'])):

                img_file = (glob.glob(os.path.join(self.dir_path, self.dataset, self.data['0'][index]) + "/" + '*.jpg')
                            + glob.glob(os.path.join(self.dir_path,
                                                     self.dataset, self.data['0'][index]) + '*.jpeg')
                            + glob.glob(os.path.join(self.dir_path, self.dataset, self.data['0'][index]) + "/" + '*.png'))

                for i in img_file:
                    self.img_file_list.append(i)

        elif self.dataset == 'test':

            img_file = (glob.glob(os.path.join(self.dir_path, self.dataset) + "/" + '*.jpg')
                        + glob.glob(os.path.join(self.dir_path,
                                                 self.dataset) + '*.jpeg')
                        + glob.glob(os.path.join(self.dir_path, self.dataset) + "/" + '*.png'))

            for i in img_file:
                self.img_file_list.append(i)

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __getitem__(self, index):

        if self.annot_format == 'csv' or self.annot_format == 'txt':
            full_filename = self.img_file_list[index]
            filename_split = full_filename.split("/")
            filename = filename_split[-1]
            category = filename_split[-2]

            for i in range(len(self.data)):
                if category == self.data['0'][i]:
                    category = self.data['1'][i]
                    break
            # make the image array value between [0,1]
            image = skimage.io.imread(full_filename)
            # image=cv2.imread(full_filename)
            image = image/255.0

            image_data = ImageData(
                filename=filename, image=image, label=category, num_classes=self.num_classes, task=self.do_task, framework=self.framework_version)
            if self.function_transforms is not None:
                for tsfm in self.function_transforms:
                    tsfm(image_data)

            elif self.built_in_transforms is not None:
                aug = Augmentation(self.built_in_transforms)
                aug(image_data)

            return image_data

        mask_category = []
        scale = None
        filename = self.img_path_list[index].split("/")[-1]

        if self.annot_format == 'yolo':
            path = self.img_path_list[index]
            self.img = self.load_image(index, path)
            dh, dw, _ = self.img.shape
            self.annot = self.load_annotations_yolo(index, dh, dw)
            mask_category = None

            self.mask = None

            self.iscrowd = None
            self.target = None

        elif self.annot_format == 'coco':
            image_info = self.coco.loadImgs(self.image_ids[index])[0]
            path = os.path.join(self.dir_path, 'images',
                                self.set_name, image_info['file_name'])

            self.img = self.load_image(index, path)
            self.annot = self.load_annotations(index)

            ann_id = self.coco.getAnnIds(imgIds=self.image_ids[index])
            coco_annotations = self.coco.loadAnns(ann_id)

            for ann in coco_annotations:
                if ann['segmentation'] is not None:
                    mask = self.coco.annToMask(ann)
                    category = ann['category_id']
                    mask_category.append([mask, category])
                    self.iscrowd = ann['iscrowd']
                    self.mask.append(mask)

                else:
                    self.iscrowd = None

            self.target["boxes"] = torch.as_tensor(
                self.box, dtype=torch.float32)
            self.target["labels"] = torch.tensor(
                self.label, dtype=torch.int64)
            self.target["masks"] = self.mask
            self.target["image_id"] = torch.tensor(
                self.image_id)
            self.target["area"] = torch.tensor(
                self.area)
            self.target["iscrowd"] = torch.tensor(
                self.iscrowd, dtype=torch.int64)
        self.image_id = [index]
        self.area = (self.box[:, 3] - self.box[:, 1]) * \
            (self.box[:, 2] - self.box[:, 0])

        image_data = ImageData(filename=filename, image=self.img,
                               annotation=self.annot,  scale=scale, masks_and_category=mask_category, target=self.target, task=self.do_task, framework=self.framework_version, bbox=self.box, bbox_label=self.label, mask=self.mask, num_classes=self.num_classes)

        if self.function_transforms is not None:
            if isinstance(self.function_transforms, list):
                for tsfm in self.function_transforms:
                    tsfm(image_data)
            else:
                self.function_transforms(image_data)

        elif self.built_in_transforms is not None:
            if isinstance(self.built_in_transforms, list):
                aug = Augmentation(self.built_in_transforms)

            else:
                aug = Augmentation([self.built_in_transforms])
            aug(image_data)
        return image_data

    def __len__(self):
        if self.annot_format == 'yolo':
            return len(self.img_path_list)
        elif self.annot_format == 'coco':
            return len(self.image_ids)
        elif self.annot_format == 'csv' or self.annot_format == 'txt':
            return len(self.img_file_list)

    def visualize(self, save_path):

        if self.annot_format == 'csv' or self.annot_format == 'txt':

            all_picture = []
            for index, image in enumerate(self.img_file_list):
                all_picture.append(self.__getitem__(index))

                filename_split = image.split("/")
                filename = filename_split[-1]

                for pic in all_picture:
                    img = pic.image
                    label = pic.label
                    # write category on the image
                    cv2.putText(img, str("Label-"+label), (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

                os.makedirs(save_path, exist_ok=True)
                save_img_path = os.path.join(save_path, filename)

                skimage.io.imsave(save_img_path, img)

        if self.annot_format == 'yolo':
            sample_list = []
            file = []
            if self.function_transforms is not None:
                for image in range(len(self.img_path_list)):
                    filename = self.img_path_list[image].split("/")[-1]
                    sample_list.append(self.__getitem__(image))
                    file.append(filename)

                    for sample in sample_list:

                        img = sample.image
                        annot = sample.annot

                        for bbox in annot:

                            label = bbox[4].astype(str)
                            x1 = int(bbox[0])
                            y1 = int(bbox[1])
                            x2 = int(bbox[2])
                            y2 = int(bbox[3])
                            bbox = [x1, y1, x2, y2]
                            draw_bbox(img, bbox, label)

                    # save to the path
                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    skimage.io.imsave(save_img_path, img)

            else:
                for img_path, ann_path in zip(self.img_path_list, self.ann_path_list):
                    img = skimage.io.imread(img_path)
                    filename = img_path.split("/")[-1]

                    dh, dw, _ = img.shape
                    fl = open(ann_path, 'r')
                    data = fl.readlines()
                    fl.close()
                    # Taken from https://stackoverflow.com/questions/64096953/how-to-convert-yolo-format-bounding-box-coordinates-into-opencv-format
                    for dt in data:
                        # Split string to float
                        c, x, y, w, h = map(float, dt.split(' '))
                        left = int((x - w / 2) * dw)
                        right = int((x + w / 2) * dw)
                        top = int((y - h / 2) * dh)
                        bottom = int((y + h / 2) * dh)
                        if left < 0:
                            left = 0
                        if right > dw - 1:
                            right = dw - 1
                        if top < 0:
                            top = 0
                        if bottom > dh - 1:
                            bottom = dh - 1
                        bbox = [left, top, right, bottom]
                        draw_bbox(img, bbox, c)

                    # save to the path
                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    skimage.io.imsave(save_img_path, img)

        elif self.annot_format == 'coco':
            if self.function_transforms is not None:
                sample_list = []
                file = []
                for image in range(len(self.img_path_list)):
                    filename = self.img_path_list[image].split("/")[-1]
                    sample_list.append(self.__getitem__(image))
                    file.append(filename)

                    for sample in sample_list:
                        img = sample.image
                        annot = sample.annot

                        for bbox in annot:

                            label = bbox[4].astype(str)
                            draw_bbox(img, bbox[:4], label)

                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    skimage.io.imsave(save_img_path, img)
            else:
                for idx in range(len(self.image_ids)):
                    image_info = self.coco.loadImgs(self.image_ids[idx])[0]
                    path = os.path.join(
                        self.img_path, 'images', self.set_name, image_info['file_name'])
                    img = self.load_image(idx, path)
                    annot = self.load_annotations(idx)
                    for bbox in annot:
                        label = self.labels[bbox[4]]
                        draw_bbox(img, bbox[:4], label)
                    filename = self.coco.loadImgs(self.image_ids[idx])[
                        0]['file_name']
                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    skimage.io.imsave(save_img_path, img)

    def load_image(self, image_index, path):

        try:
            img = skimage.io.imread(path)
            if len(img.shape) == 2:
                img = skimage.color.gray2rgb(img)
            return img.astype(np.float32) / 255.0

        except Exception as e:
            print(e)

    def load_annotations_yolo(self, index, dh, dw):

        ann = []
        box = []
        label = []
        fl = open(self.ann_path_list[index], 'r')
        for dt in fl.readlines():
            dt = dt.strip()
            # Split string to float
            c, x, y, w, h = map(float, dt.split(' '))
            left = int((x - w / 2) * dw)
            right = int((x + w / 2) * dw)
            top = int((y - h / 2) * dh)
            bottom = int((y + h / 2) * dh)
            if left < 0:
                left = 0
            if right > dw - 1:
                right = dw - 1
            if top < 0:
                top = 0
            if bottom > dh - 1:
                bottom = dh - 1

            temp_ann = [left, top, right, bottom, c]
            temp_box = [left, top, right, bottom]

            ann.append(temp_ann)
            box.append(temp_box)
            label.append(c)

        fl.close()
        self.box = np.array(box)
        self.label = c
        return np.array(ann)

    def load_annotations(self, image_index):
        # get ground truth annotations in [x1, y1, x2, y2] format
        annotations_ids = self.coco.getAnnIds(
            imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        self.box = annotations[:, :4]
        self.label = annotations[:, 4]

        return annotations

    # These two functions are so the network has every label from 0 - 80 consistently
    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def calculate_classes(self):
        for ann_file in self.ann_path_list:
            with open(ann_file) as f:
                ann = f.readline().split(" ")
                self.classes_set.add(ann[0])

    @ property
    def num_classes(self):
        if self.annot_format == 'yolo':
            return len(self.classes_set)
        elif self.annot_format == 'coco':
            return len(self.classes)
        elif self.annot_format == 'csv' or self.annot_format == 'txt':
            return len(self.data['0'])


def collater(data):

    task = data[0]._task
    framework = data[0].framework

    imgs = [image_data.image for image_data in data]
    widths = [int(image_data.shape[0]) for image_data in imgs]
    heights = [int(image_data.shape[1]) for image_data in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(
            img.shape[1]), :] = torch.tensor(img)

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    temp_padded_imgs = torch.zeros(max_width, max_height, 3)
    imgs_dic = []
    for i in range(batch_size):
        img = imgs[i]
        img = torch.tensor(img)
        img = img.permute(2, 0, 1)
        imgs_dic.append(img)

    masks_and_category = [image_data.masks_and_category for image_data in data]
    if masks_and_category[0] is not None:
        for m_and_c in masks_and_category:
            for t in m_and_c:
                new_mask = torch.zeros((max_width, max_height))
                mask = t[0]
                new_mask[:int(mask.shape[0]), :int(
                    mask.shape[1])] = torch.tensor(mask)
                t[0] = new_mask

    if task == 'detection':

        annots = [image_data.annotation for image_data in data]
        scales = [image_data.scale for image_data in data]
        if framework == 'pytorch':
            images = [torch.from_numpy(image_data.image)
                      for image_data in data]
            filenames = [image_data.filename for image_data in data]
            bboxes = [torch.tensor(image_data.bbox) for image_data in data]
            bbox_labels = [torch.tensor(image_data.bbox_label)
                           for image_data in data]

            # target = [image_data.target for image_data in data]

            target = []
            for image_data in data:
                temp_dic = {}
                temp_dic['boxes'] = torch.as_tensor(
                    image_data.bbox, dtype=torch.float32)
                temp_dic['labels'] = torch.tensor(
                    image_data.bbox_label, dtype=torch.int64)
                target.append(temp_dic)

            image_data = ImageData(image=imgs_dic,
                                   filename=filenames, masks_and_category=masks_and_category, scale=scales, target=target, bbox=bboxes, bbox_label=bbox_labels)

            return image_data
        else:
            num_classes = None
            for image_data in data:
                num_classes = image_data.num_classes
                break
            annotation_dicts = [
                image_data.annotation_dict for image_data in data]
            targets = compute_targets(
                image_group=imgs, annotations_group=annotation_dicts, num_classes=num_classes)

            return padded_imgs, targets

    elif task == 'segmentation':
        if framework == 'pytorch':
            targets = [image_data.target for image_data in data]
            for target in targets:
                masks = target['masks']
                widths = [int(image_data.shape[0]) for image_data in masks]
                heights = [int(image_data.shape[1]) for image_data in masks]
                batch_size = len(masks)
                max_width = np.array(widths).max()
                max_height = np.array(heights).max()
                padded_masks = torch.zeros(batch_size, max_width, max_height)
                for i in range(batch_size):
                    mask = masks[i]
                padded_masks[i, :int(mask.shape[0]), :int(
                    mask.shape[1])] = torch.tensor(mask)
                target['masks'] = padded_masks

            return padded_imgs, targets
        else:
            num_classes = None
            for image_data in data:
                num_classes = image_data.num_classes
                break
            annotation_dicts = [
                image_data.annotation_dict for image_data in data]
            targets = compute_targets(
                image_group=imgs, annotations_group=annotation_dicts, num_classes=num_classes)

            return padded_imgs, targets
    elif task == 'classification':
        labels = torch.LongTensor([to_categorical(
            image_data.label, num_classes=image_data.num_classes) for image_data in data])

        filenames = [image_data.filename for image_data in data]
        image_data = ImageData(
            image=padded_imgs, filename=filenames, label=labels, task=False)
        if framework == 'pytorch':
            return image_data
        else:
            return (padded_imgs, labels)


def detection_augment_list():
    l = [
        (Translate_Y, -0.3, 0.3),
        (Translate_Y_BBoxes, -0.3, 0.3),
        (Translate_X, -0.3, 0.3),
        (Translate_X_BBoxes, -0.3, 0.3),
        (CutOut, 6, 20),
        (CutOut_BBoxes, 6, 20),
        (Rotate, -30, 30),
        (ShearX, -30, 30),
        (ShearX_BBoxes, -30, 30),
        (ShearY, -30, 30),
        (ShearY_BBoxes, -30, 30),
        (Equalize, 0, 1),
        (Equalize_BBoxes, 0, 1),
        (Solarize, -1., 1.),
        (Solarize_BBoxes, -1., 1.),
        (Color, 0., 3.),
        (Color_BBoxes, 0., 3.),
        (FlipLR, 0, 1)
    ]
    return l


detection_augment_dict = {fn.__name__: (
    fn, v1, v2) for fn, v1, v2 in detection_augment_list()}


def get_augment(name, detection=False):
    if detection:
        return detection_augment_dict[name]
    else:
        pass


def apply_augment(sample, name, level, detection=False):
    augment_obj, low, high = get_augment(name, detection)
    random_add = random.random() * 0.05
    adjusted_level = (level + random_add) * (high - low) + low
    augment_inst = augment_obj(adjusted_level)
    return augment_inst(sample)


class Augmentation(object):
    def __init__(self, policies, detection=True):
        self.policies = policies
        self.detection = detection

    def __call__(self, sample):
        for _ in range(1):
            policy = random.choice(self.policies)
            print("policy: ", policy)
            for name, pr, level in [policy]:
                if random.random() > pr:
                    continue
                sample = apply_augment(sample, name, level, self.detection)

        return sample


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, min_side=608, max_side=1024):
        self.min_side = min_side
        self.max_side = max_side

    def __call__(self, sample):
        image, annots = sample[0], sample[1]

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(
            image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape
        # add padding for fpn
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros(
            (rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return torch.from_numpy(new_image), torch.from_numpy(annots), scale


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        img, annots = sample[0], sample[1]
        if np.random.rand() < flip_x:

            img = img[:, ::-1, :]

            rows, cols, channels = img.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

        return img, annots


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample.image, sample.annot

        return ((image.astype(np.float32) - self.mean) / self.std),  annots


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]


def generate_Anchors(image_shape):
    anchor_params = None
    pyramid_levels = None
    # if config and 'anchor_parameters' in config:
    #     anchor_params = parse_anchor_parameters(config)
    # if config and 'pyramid_levels' in config:
    #     pyramid_levels = parse_pyramid_levels(config)

    return anchors_for_shape(image_shape, anchor_params=anchor_params, pyramid_levels=pyramid_levels, shapes_callback=guess_shapes)


def compute_targets(image_group, annotations_group, num_classes):
    """ Compute target outputs for the network using images and their annotations.
    """
    # get the max image shape
    max_shape = tuple(max(image.shape[x]
                          for image in image_group) for x in range(3))
    anchors = generate_Anchors(max_shape)

    batches = anchor_targets_bbox(
        anchors,
        image_group,
        annotations_group,
        num_classes,
    )

    return list(batches)


class AnchorParameters:
    """ The parameteres that define how anchors are generated.
    Args
        sizes   : List of sizes to use. Each size corresponds to one feature level.
        strides : List of strides to use. Each stride correspond to one feature level.
        ratios  : List of ratios to use per location in a feature map.
        scales  : List of scales to use per location in a feature map.
    """

    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


"""
The default anchor parameters.
"""
AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=np.array([0.5, 1, 2], keras.backend.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 **
                     (2.0 / 3.0)], keras.backend.floatx()),
)


def anchor_targets_bbox(
    anchors,
    image_group,
    annotations_group,
    num_classes,
    negative_overlap=0.4,
    positive_overlap=0.5
):
    """ Generate anchor targets for bbox detection.
    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: List of annotation dictionaries with each annotation containing 'labels' and 'bboxes' of an image.
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).
    Returns
        labels_batch: batch that contains labels & anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      where N is the number of anchors for an image and the last column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch that contains bounding-box regression targets for an image & anchor states (np.array of shape (batch_size, N, 4 + 1),
                      where N is the number of anchors for an image, the first 4 columns define regression targets for (x1, y1, x2, y2) and the
                      last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """

    assert(len(image_group) == len(annotations_group)
           ), "The length of the images and annotations need to be equal."
    assert(len(annotations_group) >
           0), "No data received to compute anchor targets for."
    for annotations in annotations_group:
        assert('bboxes' in annotations), "Annotations should contain bboxes."
        assert('labels' in annotations), "Annotations should contain labels."

    batch_size = len(image_group)

    regression_batch = np.zeros(
        (batch_size, anchors.shape[0], 4 + 1), dtype=keras.backend.floatx())
    labels_batch = np.zeros(
        (batch_size, anchors.shape[0], num_classes + 1), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations['bboxes'].shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(
                anchors, annotations['bboxes'], negative_overlap, positive_overlap)

            labels_batch[index, ignore_indices, -1] = -1
            labels_batch[index, positive_indices, -1] = 1

            regression_batch[index, ignore_indices, -1] = -1
            regression_batch[index, positive_indices, -1] = 1

            # compute target class labels
            labels_batch[index, positive_indices, annotations['labels']
                         [argmax_overlaps_inds[positive_indices]].astype(int)] = 1

            regression_batch[index, :, :-1] = bbox_transform(
                anchors, annotations['bboxes'][argmax_overlaps_inds, :])

        # ignore annotations outside of image
        if image.shape:
            anchors_centers = np.vstack(
                [(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
            indices = np.logical_or(
                anchors_centers[:, 0] >= image.shape[1], anchors_centers[:, 1] >= image.shape[0])

            labels_batch[index, indices, -1] = -1
            regression_batch[index, indices, -1] = -1

    return regression_batch, labels_batch


def compute_gt_annotations(
    anchors,
    annotations,
    negative_overlap=0.4,
    positive_overlap=0.5
):
    """ Obtain indices of gt annotations with the greatest overlap.
    Args
        anchors: np.array of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (N, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).
    Returns
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """

    overlaps = compute_overlap(anchors.astype(
        np.float64), annotations.astype(np.float64))
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

    # assign "dont care" labels
    positive_indices = max_overlaps >= positive_overlap
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """

    if ratios is None:
        ratios = AnchorParameters.default.ratios

    if scales is None:
        scales = AnchorParameters.default.scales

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

    return anchors


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.
    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """

    # create a grid starting from half stride from the top left corner
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    anchor_params=None,
    shapes_callback=None,
):
    """ Generators anchors for a given shape.
    Args
        image_shape: The shape of the image.
        pyramid_levels: List of ints representing which pyramids to use (defaults to [3, 4, 5, 6, 7]).
        anchor_params: Struct containing anchor parameters. If None, default values are used.
        shapes_callback: Function to call for getting the shape of the image at different pyramid levels.
    Returns
        np.array of shape (N, 4) containing the (x1, y1, x2, y2) coordinates for the anchors.
    """

    if pyramid_levels is None:
        pyramid_levels = [3, 4, 5, 6, 7]

    if anchor_params is None:
        anchor_params = AnchorParameters.default

    if shapes_callback is None:
        shapes_callback = guess_shapes
    image_shapes = shapes_callback(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors = generate_anchors(
            base_size=anchor_params.sizes[idx],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales
        )
        shifted_anchors = shift(
            image_shapes[idx], anchor_params.strides[idx], anchors)
        all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def guess_shapes(image_shape, pyramid_levels):
    """Guess shapes based on pyramid levels.
    Args
         image_shape: The shape of the image.
         pyramid_levels: A list of what pyramid levels are used.
    Returns
        A list of image shapes at each pyramid level.
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x)
                    for x in pyramid_levels]
    return image_shapes


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """Compute bounding-box regression targets for an image."""

    # The Mean and std are calculated from COCO dataset.
    # Bounding box normalization was firstly introduced in the Fast R-CNN paper.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825  for more details
    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError(
            'Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError(
            'Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    # According to the information provided by a keras-retinanet author, they got marginally better results using
    # the following way of bounding box parametrization.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825 for more details
    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    targets = (targets - mean) / std

    return targets
