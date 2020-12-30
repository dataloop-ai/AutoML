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


class ImageData(object):

    def __init__(self, filename=None, flag=True, image=None, annotation=None, scale=None, label=None, masks_and_category=None):
        ''' filename is for saving file
            flag is for knowing the task is object detection (True) or image classification (False)
            image is the array of the image
            annotation is the bounding box , label information
            scale is the product after some of transform function should be keep
            label is the image classification's category
            masks_ad_category is a tuple ,store each mask with its category id
        '''

        self.image = image
        self.annotation = annotation
        self.filename = filename
        self.label = label

        self.flag = flag
        self.scale = scale
        self.masks_and_category = masks_and_category
        self.masks = []
        self.categories = []

        if self.flag == False:
            self.image = skimage.io.imread(self.image)

    def __str__(self):
        '''if print out the object, will call this method'''

        # choose object detection or image classification
        if self.flag:
            return "image: %s , annotation: %s , scale: %s" % (self.image, self.annotation, self.scale)
        else:
            return "filename: %s , category: %s" % (self.filename, self.label)

    def visualize(self, alpha=0.5, path=None, instance=False):
        """visualize the image with bounding box around it and mask cover it.
            alpha: how strong of the mask will be coverd on the image, between [0,1]
            path: if user input the path, not only visualize the image with segementation but also save to the path
            instance: boolean value, True for semantic segmentation false for instance segmentation
         """
        if self.flag:
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
                        self.image = self.apply_mask(
                            self.image, self.masks[i], color_cat_dict[self.categories[i]], alpha)

                else:
                    for mask in self.masks:
                        color = list(np.random.choice(
                            range(256), size=3) / 255)
                        self.image = self.apply_mask(
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
            cv2.putText(self.image, str("Label-"+self.label), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        ''' if no path given, just display the image.
            if the path is given, apply the visualization and save the image to the path
        '''

        if path is None:

            skimage.io.imshow(self.image)
            skimage.io.show()

        else:
            skimage.io.imshow(self.image)
            skimage.io.show()
            os.makedirs(path, exist_ok=True)
            save_img_path = os.path.join(path, self.filename)
            skimage.io.imsave(save_img_path, self.image)

    def apply_mask(self, image, mask, color, alpha):
        """Apply the given mask to the image with calor.
        """
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image[:, :, c])

        return image
