import math
import torch
import random
import numpy as np
import torch.nn as nn
from numpy import int64 as int64
import torchvision.transforms as transforms
from imgaug import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image, ImageOps, ImageFilter
import albumentations as A
import cv2
from .image import *
from itertools import product


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample.image
        mask = sample.annot
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        sample.image = img
        sample.annot = mask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img, mask = sample.image, sample.annotation

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        sample.image = img
        sample.annotation = mask


class RandomHorizontalFlip(object):
    def __call__(self, sample):

        p = random.random()

        if p < 0.5:
            HorizontalFlip(p=1)(sample)


class Translate_Y(object):
    def __init__(self, v):
        self.v = v  # -0.3 - 0.3 ??

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        aug = iaa.geometric.TranslateY(percent=self.v)
        img_aug = aug(image=img)
        annot_aug = []
        if annot is not None:
            bbs = BoundingBoxesOnImage(
                [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                             label=str(int(ann[4]))) for ann in annot],
                shape=img.shape)
            bbs_aug = aug(bounding_boxes=bbs)

            annot_aug = np.array(
                [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])
        segmap = []

        if masks != []:
            for mask in masks:
                segmap.append(SegmentationMapsOnImage(mask, shape=img.shape))
            mask_aug = aug(segmentation_maps=segmap)
            # reshape back to 2D
            for mask in mask_aug:
                mask.arr = mask.arr[:, :, 0]
            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_aug[i].arr

        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))

        sample.image = img_aug
        sample.annotation = annot_aug


# TODO: Enhance this function so bounding boxes will account for change in actual object,
# i.e. if the translateY bbox moves the object up, the lower limit of the bbox should move up
class Translate_Y_BBoxes(object):
    def __init__(self, v):
        self.v = v

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        unique_labels = np.unique(
            annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(
            labels=unique_labels, foreground=iaa.geometric.TranslateY(percent=self.v))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        mask_aug = []
        if masks is not None:
            for mask in masks:
                mask = SegmentationMapsOnImage(
                    mask, shape=img.shape).arr
                segmap, _ = aug(image=mask, bounding_boxes=bbs)

                mask_aug.append(segmap)
            # back to 2D array
            mask_result = []
            for mask in mask_aug:
                mask_result.append(mask[:, :, 0])

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_result[i]

        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class Translate_X(object):
    def __init__(self, v):
        self.v = v

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        aug = iaa.geometric.TranslateX(percent=self.v)
        img_aug = aug(image=img)
        annot_aug = []
        if annot is not None:
            bbs = BoundingBoxesOnImage(
                [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                             label=str(int(ann[4]))) for ann in annot],
                shape=img.shape)
            bbs_aug = aug(bounding_boxes=bbs)

            annot_aug = np.array(
                [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])
        segmap = []

        if masks != []:
            for mask in masks:
                segmap.append(SegmentationMapsOnImage(mask, shape=img.shape))
            mask_aug = aug(segmentation_maps=segmap)
            # reshape back to 2D
            for mask in mask_aug:
                mask.arr = mask.arr[:, :, 0]
            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_aug[i].arr
        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class Translate_X_BBoxes(object):
    def __init__(self, v):
        self.v = v

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        unique_labels = np.unique(
            annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(
            labels=unique_labels, foreground=iaa.geometric.TranslateX(percent=self.v))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        mask_aug = []
        if masks is not None:
            for mask in masks:
                mask = SegmentationMapsOnImage(
                    mask, shape=img.shape).arr
                segmap, _ = aug(image=mask, bounding_boxes=bbs)

                mask_aug.append(segmap)
            # back to 2D array
            mask_result = []
            for mask in mask_aug:
                mask_result.append(mask[:, :, 0])

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_result[i]

        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class CutOut(object):
    def __init__(self, v):
        self.v = v  # between 6 - 20

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.Cutout(nb_iterations=int(round(self.v)),
                         size=0.05, fill_mode="gaussian")
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class CutOut_BBoxes(object):
    def __init__(self, v):
        self.v = v  # self.v should be between 6 - 20

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        unique_labels = np.unique(
            annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.Cutout(nb_iterations=int(round(self.v)), size=0.05, fill_mode="gaussian"))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class Rotate(object):
    def __init__(self, v):
        self.v = v  # between -30 - 30

    def __call__(self, sample):
        # , sample['mask_category']
        img, annot = sample.image, sample.annotation

        annot_aug = []
        aug = iaa.Rotate(rotate=self.v)
        if annot is not None:
            bbs = BoundingBoxesOnImage(
                [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                             label=str(int(ann[4]))) for ann in annot],
                shape=img.shape)
            bbs_aug = aug(bounding_boxes=bbs)
            annot_aug = np.array(
                [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        img_aug = aug(image=img)

        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])
        segmap = []

        if masks != []:
            for mask in masks:
                segmap.append(SegmentationMapsOnImage(mask, shape=img.shape))
            mask_aug = aug(segmentation_maps=segmap)
            # reshape back to 2D

            for mask in mask_aug:
                mask.arr = mask.arr[:, :, 0]

            new_bbox = []

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_aug[i].arr

                binary_mask = np.array(index[0], np.uint8)

                contours, hierarchy = cv2.findContours(
                    binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                if contours == []:
                    continue
                areas = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    areas.append(area)

                idx = areas.index(np.max(areas))
                x, y, w, h = cv2.boundingRect(contours[idx])
                bounding_box = [x, y, x+w, y+h]

                temp = [x, y, x+w, y+h, np.float32(index[1])]
                new_bbox.append(temp)

    # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))

        sample.image = img_aug
        sample.annotation = annot_aug
        if masks != []:
            sample.annotation = np.array(new_bbox)


# TODO: Figure out how to make rotate just bboxes work correctly

# class Rotate_BBoxes(object):
#     def __init__(self):
#         pass
#
#     def __call__(self, sample):
#         img, annot = sample['img'], sample['annot']
#         unique_labels = np.unique(annot[:, 4].astype('int').astype('str')).tolist()
#
#         bbs = BoundingBoxesOnImage(
#             [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
#             shape=img.shape)
#         aug = iaa.Rotate(30)
#         img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
#         rotate_bb_aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
#                                                     foreground=iaa.Rotate(30))
#         img_rotate_bb_aug, bbs_rotate_bb_aug = rotate_bb_aug(image=img, bounding_boxes=bbs_aug)
#         drawn_img = bbs_rotate_bb_aug.draw_on_image(img_rotate_bb_aug * 255, size=2, color=[0, 255., 0])
#         import skimage
#         skimage.io.imsave('draw10.png', drawn_img)


class ShearX(object):
    def __init__(self, v):
        self.v = v  # between -30 - 30

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        aug = iaa.ShearX(self.v)
        img_aug = aug(image=img)
        annot_aug = []
        if annot is not None:
            bbs = BoundingBoxesOnImage(
                [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                             label=str(int(ann[4]))) for ann in annot],
                shape=img.shape)
            bbs_aug = aug(bounding_boxes=bbs)

            annot_aug = np.array(
                [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        mask_aug = []
        if masks != []:
            for mask in masks:
                mask = SegmentationMapsOnImage(
                    mask, shape=img.shape).arr
                segmap = aug(image=mask)

                mask_aug.append(segmap)
            # back to 2D array
            mask_result = []
            for mask in mask_aug:
                mask_result.append(mask[:, :, 0])

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_result[i]

        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class ShearX_BBoxes(object):
    def __init__(self, v):
        self.v = v  # self.v should be between -30 - 30

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation

        unique_labels = np.unique(
            annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.ShearX(self.v))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        mask_aug = []
        if masks is not None:
            for mask in masks:
                mask = SegmentationMapsOnImage(
                    mask, shape=img.shape).arr
                segmap, _ = aug(image=mask, bounding_boxes=bbs)

                mask_aug.append(segmap)
            # back to 2D array
            mask_result = []
            for mask in mask_aug:
                mask_result.append(mask[:, :, 0])

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_result[i]

        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class ShearY(object):
    def __init__(self, v):
        self.v = v  # between -30 - 30

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        aug = iaa.ShearY(self.v)
        img_aug = aug(image=img)
        annot_aug = []
        if annot is not None:

            bbs = BoundingBoxesOnImage(
                [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                             label=str(int(ann[4]))) for ann in annot],
                shape=img.shape)
            bbs_aug = aug(bounding_boxes=bbs)
            annot_aug = np.array(
                [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        mask_aug = []
        if masks != []:
            for mask in masks:
                mask = SegmentationMapsOnImage(
                    mask, shape=img.shape).arr
                segmap = aug(image=mask)

                mask_aug.append(segmap)
            # back to 2D array
            mask_result = []
            for mask in mask_aug:
                mask_result.append(mask[:, :, 0])

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_result[i]

        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class ShearY_BBoxes(object):
    def __init__(self, v):
        self.v = v  # self.v should be between -30 - 30

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        unique_labels = np.unique(
            annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.ShearY(self.v))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        mask_aug = []
        if masks is not None:
            for mask in masks:
                mask = SegmentationMapsOnImage(
                    mask, shape=img.shape).arr
                segmap, _ = aug(image=mask, bounding_boxes=bbs)

                mask_aug.append(segmap)
            # back to 2D array
            mask_result = []
            for mask in mask_aug:
                mask_result.append(mask[:, :, 0])

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_result[i]

        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class Equalize(object):
    def __init__(self, v):
        self.v = v  # not applied

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.AllChannelsHistogramEqualization()
        img_aug, bbs_aug = aug(
            image=(img * 255.).astype('uint8'), bounding_boxes=bbs)
        img_aug = img_aug.astype('float32') / 255.
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class Equalize_BBoxes(object):
    def __init__(self, v):
        self.v = v  # not applied

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        unique_labels = np.unique(
            annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.AllChannelsHistogramEqualization())
        img_aug, bbs_aug = aug(
            image=(img * 255.).astype('uint8'), bounding_boxes=bbs)
        img_aug = img_aug.astype('float32') / 255.
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class Solarize(object):
    def __init__(self, v):
        self.v = v  # -1 - 1.

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.pillike.Solarize(threshold=self.v)
        img_aug, bbs_aug = aug(image=(img * 2. - 1.), bounding_boxes=bbs)
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        img_aug = (img_aug + 1.) / 2
        sample.image = img_aug
        sample.annotation = annot_aug


class Solarize_BBoxes(object):
    def __init__(self, v):
        self.v = v  # -1 - 1

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        unique_labels = np.unique(
            annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.pillike.Solarize(threshold=0.))
        img_aug, bbs_aug = aug(image=(img * 2. - 1.), bounding_boxes=bbs)
        img_aug = (img_aug + 1.) / 2
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class Color(object):
    def __init__(self, v):
        self.v = v  # 0.0 - 3.0

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.pillike.EnhanceColor(self.v)
        img_aug, bbs_aug = aug(
            image=(img * 255.).astype('uint8'), bounding_boxes=bbs)
        img_aug = img_aug.astype('float32') / 255.
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class Color_BBoxes(object):
    def __init__(self, v):
        self.v = v  # not applied?

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        unique_labels = np.unique(
            annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.pillike.EnhanceColor(self.v))
        img_aug, bbs_aug = aug(
            image=(img * 255.).astype('uint8'), bounding_boxes=bbs)
        img_aug = img_aug.astype('float32') / 255.
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        unique_labels = np.unique(
            annot[:, 4].astype('int').astype('str')).tolist()

        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(
            labels=unique_labels, foreground=iaa.geometric.TranslateY(percent=0.1))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        # drawn_img = bbs_aug.draw_on_image(img_aug * 255, size=2, color=[0, 255., 0])
        # import skimage
        # skimage.io.imsave('draw.png', drawn_img)
        # img = img.rotate(rotate_degree, Image.BILINEAR)
        # mask = mask.rotate(rotate_degree, Image.NEAREST)

        sample.image = img_aug
        sample.annotation = bbs_aug

# FLIP LR BBOXES ONLY DOESNT SEEM to WORK WITH THIS LIBRARY SO FAR


class FlipLR(object):
    def __init__(self, v):
        self.v = v  # ignore ??

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                         label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.Fliplr()
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array(
            [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        sample.image = img_aug
        sample.annotation = annot_aug


# TODO: fix this later
class RandomGaussianBlur(object):
    def __init__(self, v):
        self.v = v  # 0.0 - 3.0

    def __call__(self, sample):

        if random.random() < 0.5:
            GaussianBlur(self.v)(sample)


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img, mask = sample.image, sample.annotation
        # random scale (short edge)
        short_size = random.randint(
            int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(
                0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        sample.image = img
        sample.annotation = mask


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img, mask = sample.image, sample.annotation
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        sample.image = img
        sample.annotation = mask


# resize to 512*1024
class FixedResize(object):
    """change the short edge length to size"""

    def __init__(self, resize=512):
        self.size1 = resize  # size= 512

    def __call__(self, sample):
        img, mask = sample.image, sample.annotaiton
        assert img.size == mask.size

        w, h = img.size
        if w > h:
            oh = self.size1
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.size1
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        sample.image = img
        sample.annotation = mask

# TODO fix bounding box
# random crop 321*321


class RandomCrop(object):
    def __init__(self, crop_size=320):
        self.crop_size = crop_size

    def __call__(self, sample):
        # img, mask = sample.image, sample.annotation
        # w, h, _ = img.shape
        # x1 = random.randint(0, w - self.crop_size)
        # y1 = random.randint(0, h - self.crop_size)
        # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # sample.image = img
        # sample.annotation = mask
        img, annot = sample.image, sample.annotation

        aug = A.RandomCrop(self.crop_size, self.crop_size)

        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        image_crop = []
        bbox_crop = []
        if masks is not None:
            augmented = aug(image=img, bboxes=annot, masks=masks)
            mask_crop = augmented['masks']
            image_crop = augmented['image']
            bbox_crop = augmented['bboxes']

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_crop[i]
        else:
            augmented = aug(image=img, bboxes=annot)
            image_crop = augmented['image']
            bbox_crop = augmented['bboxes']

        # the shape has to be at least (0,5)
        if len(bbox_crop) == 0:
            bbox_crop = np.zeros((0, 5))

        sample.image = image_crop
        sample.annotation = bbox_crop


class RandomScale(object):
    def __init__(self, scales=0.5):
        self.scales = scales  # float

    def __call__(self, sample):

        # img, mask = sample.image, sample.annotation
        # w, h = img.size
        # scale = random.choice(self.scales)
        # w, h = int(w * scale), int(h * scale)

        # sample.image = img
        # sample.annotation = mask

        img, annot = sample.image, sample.annotation

        aug = A.RandomScale(scale_limit=self.scales)

        bbox_aug = []
        if annot is not None:
            augmented = aug(image=img, bboxes=annot)
            bbox_aug = augmented['bboxes']
        else:
            augmented = aug(image=img)
        image_aug = augmented['image']

        # the shape has to be at least (0,5)
        if len(bbox_aug) == 0:
            bbox_aug = np.zeros((0, 5))

        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        if masks != []:
            w, h, _ = image_aug.shape

            mask_aug = []
            if masks is not None:
                for mask in masks:
                    mask = cv2.resize(mask, (h, w))
                    mask_aug.append(mask)

                for i, index in enumerate(sample.masks_and_category):
                    index[0] = mask_aug[i]

        sample.image = image_aug
        sample.annotation = bbox_aug


class TransformTr(object):
    def __init__(self, resize, multi_scale=None):
        if multi_scale is None:
            self.composed_transforms = transforms.Compose([
                FixedResize(resize=resize),
                # RandomCrop(crop_size=args.crop_size),
                # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
                # tr.RandomGaussianBlur(),
                # Normalize(mean, std),
                # ToTensor()
            ])
        else:
            self.composed_transforms = transforms.Compose([
                FixedResize(resize=args.resize),
                RandomScale(scales=args.multi_scale),
                RandomCrop(crop_size=args.crop_size),
                # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
                # tr.RandomGaussianBlur(),
                Normalize(mean, std),
                ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)


class TransformVal(object):
    def __init__(self, args, mean, std):
        self.composed_transforms = transforms.Compose([
            FixedResize(resize=args.resize),
            FixScaleCrop(crop_size=args.crop_size),  # TODO:CHECK THIS
            Normalize(mean, std),
            ToTensor()])

    def __call__(self, sample):
        return self.composed_transforms(sample)


class HorizontalFlip(object):
    def __init__(self, p):
        self.p = p  # probability between 0-1

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation

        aug = A.HorizontalFlip(p=self.p)
        augmented = aug(image=img)
        image_h_flipped = augmented['image']
        bbox_h_flipped = []
        if annot is not None:
            augmented = aug(image=img, bboxes=annot)
            bbox_h_flipped = augmented['bboxes']

        # the shape has to be at least (0,5)
        if len(bbox_h_flipped) == 0:
            bbox_h_flipped = np.zeros((0, 5))

        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        if masks != []:
            augmented_mask = aug(image=img, masks=masks)
            mask_h_flipped = augmented_mask['masks']

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_h_flipped[i]

        sample.image = image_h_flipped
        sample.annotation = bbox_h_flipped


class VerticalFlip(object):
    def __init__(self, p):
        self.p = p  # probability between 0-1

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation

        aug = A.VerticalFlip(p=self.p)

        augmented = aug(image=img)
        image_v_flipped = augmented['image']
        bbox_v_flipped = []
        if annot is not None:
            augmented = aug(image=img, bboxes=annot)
            bbox_v_flipped = augmented['bboxes']

        # the shape has to be at least (0,5)
        if len(bbox_v_flipped) == 0:
            bbox_v_flipped = np.zeros((0, 5))

        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        if masks != []:
            augmented_mask = aug(image=img, masks=masks)
            mask_v_flipped = augmented_mask['masks']

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_v_flipped[i]

        sample.image = image_v_flipped
        sample.annotation = bbox_v_flipped


class GaussianBlur(object):
    def __init__(self, v):
        self.v = v  # 0.0 - 3.0

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        aug = iaa.GaussianBlur(sigma=self.v)
        img_aug = aug(image=img)
        annot_aug = []
        if annot is not None:
            bbs = BoundingBoxesOnImage(
                [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                             label=str(int(ann[4]))) for ann in annot],
                shape=img.shape)
            bbs_aug = aug(bounding_boxes=bbs)
            annot_aug = np.array(
                [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        # the shape has to be at least (0,5)

        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class MotionBlur(object):
    def __init__(self, k):
        self.k = k  # 0.0 - 15.0

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        aug = iaa.MotionBlur(k=self.k)
        img_aug = aug(image=img)
        annot_aug = []
        if annot is not None:
            bbs = BoundingBoxesOnImage(
                [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                             label=str(int(ann[4]))) for ann in annot],
                shape=img.shape)
            bbs_aug = aug(bounding_boxes=bbs)

            annot_aug = np.array(
                [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class ElasticTransformation(object):
    def __init__(self, alpha):
        self.alpha = alpha     # 10,20,30,...,100
        self.sigma = self.alpha/10

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation
        aug = iaa.ElasticTransformation(alpha=self.alpha, sigma=self.sigma)
        img_aug = aug(image=img)
        annot_aug = []
        if annot is not None:
            bbs = BoundingBoxesOnImage(
                [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3],
                             label=str(int(ann[4]))) for ann in annot],
                shape=img.shape)
            bbs_aug = aug(bounding_boxes=bbs)
            annot_aug = np.array(
                [[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        # the shape has to be at least (0,5)
        if len(annot_aug) == 0:
            annot_aug = np.zeros((0, 5))
        sample.image = img_aug
        sample.annotation = annot_aug


class CenterCrop(object):
    def __init__(self, square):
        self.square = square

    def __call__(self, sample):
        img, annot = sample.image, sample.annotation

        aug = A.CenterCrop(height=self.square, width=self.square)

        augmented = aug(image=img)
        image_c_crop = augmented['image']
        bbox_c_crop = []
        if annot is not None:
            augmented = aug(image=img, bboxes=annot)
            bbox_c_crop = augmented['bboxes']

        # the shape has to be at least (0,5)
        if len(bbox_c_crop) == 0:
            bbox_c_crop = np.zeros((0, 5))

        masks = []
        if sample.masks_and_category is not None:
            for index in sample.masks_and_category:
                masks.append(index[0])

        if masks != []:
            augmented_mask = aug(image=img, masks=masks)
            mask_c_crop = augmented_mask['masks']

            for i, index in enumerate(sample.masks_and_category):
                index[0] = mask_c_crop[i]

        sample.image = image_c_crop
        sample.annotation = bbox_c_crop
