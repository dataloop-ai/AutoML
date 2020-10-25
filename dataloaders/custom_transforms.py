import math
import torch
import random
import numpy as np
import torch.nn as nn
from numpy import int64 as int64
import torchvision.transforms as transforms
from imgaug import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
from PIL import Image, ImageOps, ImageFilter


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
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class Translate_Y(object):
    def __init__(self, v):
        self.v = v # -0.3 - 0.3 ??

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.geometric.TranslateY(percent=self.v)
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}

#TODO: Enhance this function so bounding boxes will account for change in actual object,
# i.e. if the translateY bbox moves the object up, the lower limit of the bbox should move up
class Translate_Y_BBoxes(object):
    def __init__(self, v):
        self.v = v

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']
        unique_labels = np.unique(annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels, foreground=iaa.geometric.TranslateY(percent=self.v))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class Translate_X(object):
    def __init__(self, v):
        self.v = v

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.geometric.TranslateX(percent=self.v)
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class Translate_X_BBoxes(object):
    def __init__(self, v):
        self.v = v

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']
        unique_labels = np.unique(annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels, foreground=iaa.geometric.TranslateX(percent=self.v))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class CutOut(object):
    def __init__(self, v):
        self.v = v # between 6 - 20

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.Cutout(nb_iterations=int(round(self.v)), size=0.05, fill_mode="gaussian")
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class CutOut_BBoxes(object):
    def __init__(self, v):
        self.v = v #self.v should be between 6 - 20

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']
        unique_labels = np.unique(annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.Cutout(nb_iterations=int(round(self.v)), size=0.05, fill_mode="gaussian"))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class Rotate(object):
    def __init__(self, v):
        self.v = v # between -30 - 30

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.Rotate(rotate=self.v)
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}

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
        self.v = v # between -30 - 30

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.ShearX(self.v)
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class ShearX_BBoxes(object):
    def __init__(self, v):
        self.v = v #self.v should be between -30 - 30

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']
        unique_labels = np.unique(annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.ShearX(self.v))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class ShearY(object):
    def __init__(self, v):
        self.v = v # between -30 - 30

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.ShearY(self.v)
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class ShearY_BBoxes(object):
    def __init__(self, v):
        self.v = v #self.v should be between -30 - 30

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']
        unique_labels = np.unique(annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.ShearY(self.v))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class Equalize(object):
    def __init__(self, v):
        self.v = v # not applied

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.AllChannelsHistogramEqualization()
        img_aug, bbs_aug = aug(image=(img * 255.).astype('uint8'), bounding_boxes=bbs)
        img_aug = img_aug.astype('float32') / 255.
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class Equalize_BBoxes(object):
    def __init__(self, v):
        self.v = v #not applied

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']
        unique_labels = np.unique(annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.AllChannelsHistogramEqualization())
        img_aug, bbs_aug = aug(image=(img * 255.).astype('uint8'), bounding_boxes=bbs)
        img_aug = img_aug.astype('float32') / 255.
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class Solarize(object):
    def __init__(self, v):
        self.v = v # -1 - 1.

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.pillike.Solarize(threshold=self.v)
        img_aug, bbs_aug = aug(image=(img * 2. - 1.), bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])
        img_aug = (img_aug + 1.) / 2
        return {'img': img_aug, 'annot': annot_aug}


class Solarize_BBoxes(object):
    def __init__(self, v):
        self.v = v #-1 - 1

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']
        unique_labels = np.unique(annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.pillike.Solarize(threshold=0.))
        img_aug, bbs_aug = aug(image=(img * 2. - 1.), bounding_boxes=bbs)
        img_aug = (img_aug + 1.) / 2
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class Color(object):
    def __init__(self, v):
        self.v = v # 0.0 - 3.0

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.pillike.EnhanceColor(self.v)
        img_aug, bbs_aug = aug(image=(img * 255.).astype('uint8'), bounding_boxes=bbs)
        img_aug = img_aug.astype('float32') / 255.
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class Color_BBoxes(object):
    def __init__(self, v):
        self.v = v #not applied?

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']
        unique_labels = np.unique(annot[:, 4].astype('int').astype('str')).tolist()

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels,
                                          foreground=iaa.pillike.EnhanceColor(self.v))
        img_aug, bbs_aug = aug(image=(img * 255.).astype('uint8'), bounding_boxes=bbs)
        img_aug = img_aug.astype('float32') / 255.
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']
        unique_labels = np.unique(annot[:, 4].astype('int').astype('str')).tolist()

        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.BlendAlphaBoundingBoxes(labels=unique_labels, foreground=iaa.geometric.TranslateY(percent=0.1))
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        # drawn_img = bbs_aug.draw_on_image(img_aug * 255, size=2, color=[0, 255., 0])
        # import skimage
        # skimage.io.imsave('draw.png', drawn_img)
        # img = img.rotate(rotate_degree, Image.BILINEAR)
        # mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}

# FLIP LR BBOXES ONLY DOESNT SEEM to WORK WITH THIS LIBRARY SO FAR
class FlipLR(object):
    def __init__(self, v):
        self.v = v #ignore ??

    def __call__(self, sample):
        img, annot = sample['img'], sample['annot']

        bbs = BoundingBoxesOnImage(
            [BoundingBox(x1=ann[0], y1=ann[1], x2=ann[2], y2=ann[3], label=str(int(ann[4]))) for ann in annot],
            shape=img.shape)
        aug = iaa.Fliplr()
        img_aug, bbs_aug = aug(image=img, bounding_boxes=bbs)
        annot_aug = np.array([[bb.x1, bb.y1, bb.x2, bb.y2, np.float32(bb.label)] for bb in bbs_aug])

        return {'img': img_aug, 'annot': annot_aug}

# TODO: fix this later
class RandomGaussianBlur(object):
    def __init__(self, _):
        pass

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
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
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
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

        return {'image': img,
                'label': mask}


# resize to 512*1024
class FixedResize(object):
    """change the short edge length to size"""

    def __init__(self, resize=512):
        self.size1 = resize  # size= 512

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
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
        return {'image': img,
                'label': mask}


# random crop 321*321
class RandomCrop(object):
    def __init__(self, crop_size=320):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': img,
                'label': mask}


class RandomScale(object):
    def __init__(self, scales=(1,)):
        self.scales = scales

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        scale = random.choice(self.scales)
        w, h = int(w * scale), int(h * scale)
        return {'image': img,
                'label': mask}


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
