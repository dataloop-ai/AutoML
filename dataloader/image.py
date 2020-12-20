import os

import skimage

from .utils import draw_bbox
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class ImageData(object):

    def __init__(self,img, annot, scale=None):
        self.image = img
        self.annot = annot
        self.scale = scale

    def __str__(self):
        return "img: #%s, annot: %s, scale: %s" % (self.img, self.annot, self.scale)

    def visualize(self, path=None):


        for bbox in self.annot:
            label = bbox[4]
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])

            b_list = [x1, y1, x2, y2]
            cv2.rectangle(self.image, (x1, y1), (x2, y2), color=(0, 0, 1), thickness=2)
            

        # b, g, r = cv2.split(self.img)  # get b, g, r
        # rgb_img1 = cv2.merge([r, g, b])  # switch it to r, g, b
        if path is None:

            plt.imshow(self.image)
            plt.show()
        else:
            plt.imshow(self.image)
            plt.show()
            os.makedirs(path, exist_ok=True)

            skimage.io.imsave(path, self.image)



