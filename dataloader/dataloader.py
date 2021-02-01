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
    def __init__(self, dir_path, annot_format, do_task=None, framework_version=None, num_categories=None, annotation_path=None, function_transforms=None, built_in_transforms=None, dataset="train"):
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
        self.num_categories = num_categories
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
                filename=filename, image=image, label=category, num_classes=self.num_categories, task=self.do_task, framework=self.framework_version)
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

            self.target["boxes"] = self.box
            self.target["labels"] = self.label
            self.target["masks"] = self.mask
            self.target["image_id"] = self.image_id
            self.target["area"] = self.area
            self.target["iscrowd"] = self.iscrowd

        self.image_id = [index]
        self.area = (self.box[:, 3] - self.box[:, 1]) * \
            (self.box[:, 2] - self.box[:, 0])

        image_data = ImageData(filename=filename, image=self.img,
                               annotation=self.annot,  scale=scale, masks_and_category=mask_category, target=self.target, task=self.do_task, framework=self.framework_version)

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

    task = 'detection'
    for image_data in data:
        if image_data._task == 'detection':
            break
        elif image_data._task == 'segmentation':
            task = 'segmentation'
            break
        else:
            task = 'classification'
            break

    framework = 'pytorch'
    for image_data in data:
        if image_data.framework == 'keras':
            framework = 'keras'
            break

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

    if framework == 'keras':
        padded_imgs = np.array(padded_imgs)
    else:
        padded_imgs = padded_imgs.permute(0, 3, 1, 2)

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

        # max_num_annots = max(annot.shape[0] for annot in annots)
        max_num_annots = 0
        for annot in annots:
            if annot is not None:
                if annot.shape[0] > max_num_annots:
                    max_num_annots = annot.shape[0]

        if max_num_annots > 0:

            annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

            if max_num_annots > 0:
                for idx, annot in enumerate(annots):
                    # print(annot.shape)
                    if annot.shape[0] > 0:
                        annot_padded[idx, :annot.shape[0],
                                     :] = torch.tensor(annot)
        else:
            annot_padded = torch.ones((len(annots), 1, 5)) * -1

        filenames = [image_data.filename for image_data in data]

        image_data = ImageData(image=padded_imgs, annotation=annot_padded,
                               filename=filenames, masks_and_category=masks_and_category, scale=scales)
        return image_data

    elif task == 'segmentation':

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
