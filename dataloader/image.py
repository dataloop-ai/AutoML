import colorsys
import os
import random
import skimage
from skimage import draw
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pycocotools import coco
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from skimage import io
import torch


class ImageData(object):

    def __init__(self, num_classes=None, filename=None, task=None, image=None, annotation=None, scale=None, label=None, masks_and_category=None, target=None, framework=None, bbox=None, bbox_label=None, mask=None):
        ''' filename is for saving file
            _task is for knowing the task is object detection  or image classification or instance segmentation
            image is the array of the image #TODO: CHANGE TO IMAGES
            annotation is the bounding box , label information
            scale is the product after some of transform function should be keep
            label is the image classification's category
            masks_ad_category is a tuple ,store each mask with its category id
        '''

        self.image = image
        self.annotation = annotation
        self.filename = filename
        self.label = label

        # TODO: CHANGE THIS to self._task #Underscore everything thats internal

        self.scale = scale
        self.masks_and_category = masks_and_category
        self.masks = []
        self.categories = []
        self.num_classes = num_classes
        self._task = task
        self.framework = framework

        self.bbox = bbox
        self.bbox_label = bbox_label
        self.mask = mask

        self.target = target

        self.annotation_dict = {}
        if self.bbox is not None:
            self.annotation_dict['bboxes'] = self.bbox
        if self.bbox_label is not None:
            self.annotation_dict['labels'] = self.bbox_label

    def __str__(self):
        '''if print out the object, will call this method'''

        # choose object detection or image classification
        if self._task == 'detection':
            return "image: %s , annotation: %s , scale: %s" % (self.image, self.annotation, self.scale)
        else:
            return "filename: %s , category: %s" % (self.filename, self.label)

    def visualize(self, alpha=0.5, path=None, instance=False):
        """visualize the image with bounding box around it and mask cover it.
            alpha: how strong of the mask will be coverd on the image, between [0,1]
            path: if user input the path, not only visualize the image with segementation but also save to the path
            instance: boolean value, True for semantic segmentation false for instance segmentation
            # TODO: MAKE IT WORK FOR IMAGES PLURAL
         """
        if self._task == 'dectection':
            if isinstance(self.filename, list):
                temp_img = list(self.image)
                for i in range(len(self.filename)):
                    image = temp_img[i]
                    reshape_image = image.permute(1, 2, 0).numpy()
                    masks = []
                    categories = []
                    categories_set = []
                    colors = []
                    if self.masks_and_category[i] is not None:

                        for index in self.masks_and_category[i]:
                            masks.append(index[0])
                            categories.append(index[1])
                    if masks is not None:
                        if instance:

                            categories_set = set(categories)
                            for _ in range(len(categories_set)):
                                colors.append(
                                    list(np.random.choice(range(256), size=3) / 255))

                            categories_set = list(categories_set)
                            color_cat_dict = dict(zip(categories_set, colors))

                            for j in range(len(masks)):
                                reshape_image = self._apply_mask(
                                    reshape_image, masks[j], color_cat_dict[categories[j]], alpha)
                        else:
                            for mask in masks:
                                color = list(np.random.choice(
                                    range(256), size=3) / 255)
                                reshape_image = self._apply_mask(
                                    reshape_image, mask, color, alpha)
                    for bbox in self.annotation[i]:
                        label = bbox[4]
                        x1 = int(bbox[0])
                        y1 = int(bbox[1])
                        x2 = int(bbox[2])
                        y2 = int(bbox[3])

                        cv2.rectangle(reshape_image, (x1, y1), (x2, y2),
                                      color=(0, 0, 1), thickness=2)

            else:

                if self.masks_and_category is not None:
                    for index in self.masks_and_category:
                        self.masks.append(index[0])
                        self.categories.append(index[1])

                if self.masks is not None:
                    if instance:
                        colors = []
                        categories_set = set(self.categories)
                        for _ in range(len(categories_set)):
                            colors.append(
                                list(np.random.choice(range(256), size=3) / 255))

                        categories_set = list(categories_set)
                        color_cat_dict = dict(zip(categories_set, colors))

                        for i in range(len(self.masks)):
                            self.image = self._apply_mask(
                                self.image, self.masks[i], color_cat_dict[self.categories[i]], alpha)

                    else:
                        for mask in self.masks:
                            color = list(np.random.choice(
                                range(256), size=3) / 255)
                            self.image = self._apply_mask(
                                self.image, mask, color, alpha)

                for bbox in self.annotation:
                    label = bbox[4]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])

                    cv2.rectangle(self.image, (x1, y1), (x2, y2),
                                  color=(0, 0, 1), thickness=2)
        else:
            '''image classification task
                write down the category label on the image
            '''
            if isinstance(self.filename, list):
                temp_img = list(self.image)
                for i in range(len(self.filename)):
                    img = temp_img[i]
                    print(img.shape)
                    # c,h,w= img.shape
                    # reshape_image=np.resize(img,(w,h,c))
                    reshape_image = img.permute(1, 2, 0).numpy()
                    print(reshape_image.shape)
                    cv2.putText(reshape_image, str(self.label[i]), (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

            else:

                cv2.putText(self.image, str("Label-"+self.label), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        ''' if no path given, just display the image.
            if the path is given, apply the visualization and save the image to the path
        '''

        if path is None:
            if isinstance(self.filename, list):
                temp_img = list(self.image)
                for i in range(len(self.filename)):
                    img = temp_img[i]
                    print(img.shape)
                    reshape_image = img.permute(1, 2, 0).numpy()
                    print(reshape_image.shape)
                    skimage.io.imshow(reshape_image)
                    skimage.io.show()
            else:
                # IF PATH SAVE IT
                skimage.io.imshow(self.image)
                skimage.io.show()
                # TODO: PRINT WHAT YOU ARE SAVING TOO

        else:
            if isinstance(self.filename, list):
                temp_img = list(self.image)
                for i in range(len(self.filename)):
                    img = temp_img[i]
                    reshape_image = img.permute(1, 2, 0).numpy()
                    skimage.io.imshow(reshape_image)
                    skimage.io.show()
                    os.makedirs(path, exist_ok=True)
                    save_img_path = os.path.join(path, self.filename[i])
                    skimage.io.imsave(save_img_path, reshape_image)
                    print(self.filename[i], " is saving to ",
                          save_img_path, " ...")
            else:
                skimage.io.imshow(self.image)
                skimage.io.show()
                os.makedirs(path, exist_ok=True)
                save_img_path = os.path.join(path, self.filename)
                skimage.io.imsave(save_img_path, self.image)

    def _apply_mask(self, image, mask, color, alpha):
        """Apply the given mask to the image with color.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])

        return image
