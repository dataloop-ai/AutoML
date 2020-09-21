import json
import skimage
from PIL import Image
import os
from pycocotools.coco import COCO
import numpy as np


def change_coco_image_sizes_with_annotations(path_to_dataset, min_side):
    os.chdir(path_to_dataset)

    folder_paths = ['train2017', 'val2017']
    for folder_path in folder_paths:
        images_folder_path = os.path.join(os.getcwd(), 'images', folder_path)
        annotations_path = os.path.join(os.getcwd(), 'annotations', 'instances_' + folder_path + '.json')
        coco = COCO(annotations_path)
        for img_key, item in coco.imgs.items():
            img_path = os.path.join(images_folder_path, item['file_name'])
            img = Image.open(img_path)
            width, height = img.size
            smallest_side = min(width, height)
            scale = min_side / smallest_side

            img = img.resize((int(scale * width), int(scale * height)), Image.ANTIALIAS)
            [x for x in coco.dataset['images'] if x['id'] == img_key][0]['width'] = int(scale * width)
            [x for x in coco.dataset['images'] if x['id'] == img_key][0]['height'] = int(scale * height)
            img.save(img_path)
            print(scale)
            anns = coco.imgToAnns[img_key]
            for ann in anns:
                bbox = ann['bbox']
                ann_id = ann['id']
                new_bbox = np.array(bbox) * scale
                [x for x in coco.dataset['annotations'] if x['id'] == ann_id][0]['bbox'] = new_bbox.tolist()
                print(bbox)
                print([x for x in coco.dataset['annotations'] if x['id'] == ann_id][0]['bbox'])

        new_annotations_path = os.path.join(os.getcwd(), 'annotations', 'instances_large_' + folder_path + '.json')

        with open(new_annotations_path, 'w') as fp:
            json.dump(coco.dataset, fp)


change_coco_image_sizes_with_annotations('/Users/noam/tiny_coco', min_side=2464)