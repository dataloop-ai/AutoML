# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from dataloaders.custom_transforms import *
from torchvision.transforms.transforms import Compose

random_mirror = True


def detection_augment_list():
    l = [
        (Translate_Y, -0.3, 0.3),
        (Translate_Y_BBoxes, -0.3, 0.3),
        (Translate_X, -0.3, 0.3),
        (Translate_X_BBoxes, -0.3, 0.3),
        (CutOut, 6., 20.),
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

detection_augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in detection_augment_list()}

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
    return augment_inst(sample.copy())

