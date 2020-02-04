import os
import json
import numpy as np
from shutil import copyfile
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def convert_dataloop_to_coco(path_to_data, name='train', split_val=False, split_percentage=0.005):
    logger.info('converting dataloop data to coco')
    path_to_dataloop_images_dir = os.path.join(path_to_data, 'items')
    path_to_dataloop_annotations_dir = os.path.join(path_to_data, 'json')

    imgs_ls = os.listdir(path_to_dataloop_images_dir)
    num_imgs = len(imgs_ls)
    val_ind = np.random.choice(num_imgs, int(split_percentage * num_imgs))
    images = []
    images_val = []
    val_filenames = []
    for i, filename in enumerate(imgs_ls):
        with Image.open(os.path.join(path_to_dataloop_images_dir, filename)) as img:
            width, height = img.size
        if split_val:
            if i in val_ind:
                images_val.append({'file_name': filename,
                                   'id': i,
                                   'width': width,
                                   'height': height
                                   })
                val_filenames.append(filename)
                continue
        images.append({'file_name': filename,
                       'id': i,
                       'width': width,
                       'height': height
                       })
    all_images = images + images_val

    paths_to_dataloop_annotations = [os.path.join(path_to_dataloop_annotations_dir, j) for j in
                                     os.listdir(path_to_dataloop_annotations_dir) if 'json' in j]

    dataloop_jsons = []
    for json_path in paths_to_dataloop_annotations:
        with open(json_path) as jf:
            dataloop_json = json.load(jf)
        dataloop_jsons.append(dataloop_json)
    # compute labels to id
    labels = []
    for single_dataloop_json in dataloop_jsons:
        for annotation in single_dataloop_json['annotations']:
            label = annotation['label']
            labels.append(label)
    np_labels = np.array(labels)
    class_list = np.unique(np_labels)

    label_to_id = {name: i for i, name in enumerate(class_list) if name is not "done"}
    categories = [{'id': i, 'name': name} for name, i in label_to_id.items()]

    # compute annotations
    index = 0
    annotations = []
    annotations_val = []
    for single_dataloop_json in dataloop_jsons:
        filename = single_dataloop_json['filename']
        # find which is the right image and extract the id
        for img in all_images:
            if img['file_name'] in filename:
                img_id = img['id']
                break

        for annotation in single_dataloop_json['annotations']:
            try:
                x = annotation['coordinates'][0]['x']
                y = annotation['coordinates'][0]['y']
                w = annotation['coordinates'][1]['x'] - x
                h = annotation['coordinates'][1]['y'] - y
                label = annotation['label']
                index += 1
                annot = {'bbox': [x, y, w, h],
                         'category_id': label_to_id[label],
                         'image_id': img_id,
                         'iscrowd': 0,
                         'id': index
                         }
                if split_val:
                    if img_id in val_ind:
                        annotations_val.append(annot)
                        continue
                annotations.append(annot)
            except:
                print('annotation_type : ', annotation['type'], '\t annotation_label: ', annotation['label'],
                      ' \t did not convert')

    coco_json = {'images': images,
                 'annotations': annotations,
                 'categories': categories}
    if split_val:
        coco_val_json = {'images': images_val,
                         'annotations': annotations_val,
                         'categories': categories}
    logger.info(os.listdir(path_to_data))
    save_annotation_path = os.path.join(path_to_data, 'annotations', 'instances_' + name + '.json')
    if split_val:
        save_annotation_val_path = os.path.join(path_to_data, 'annotations', 'instances_' + 'val' + '.json')
    if not os.path.exists(os.path.join(path_to_data, 'annotations')):
        os.mkdir(os.path.join(path_to_data, 'annotations'))
    logger.info(os.listdir(os.path.join(path_to_data, 'annotations')))
    with open(save_annotation_path, 'w') as outfile:
        json.dump(coco_json, outfile)
    if split_val:
        with open(save_annotation_val_path, 'w') as outfile:
            json.dump(coco_val_json, outfile)
    if not os.path.exists(os.path.join(os.path.join(path_to_data, 'images'))):
        os.mkdir(os.path.join(path_to_data, 'images'))
    os.mkdir(os.path.join(path_to_data, 'images', name))
    if split_val:
        os.mkdir(os.path.join(path_to_data, 'images', 'val'))
    for img in os.listdir(os.path.join(path_to_data, 'items')):
        if split_val:
            if img in val_filenames:
                copyfile(os.path.join(path_to_data, 'items', img), os.path.join(path_to_data, 'images', 'val', img))
                continue
        copyfile(os.path.join(path_to_data, 'items', img), os.path.join(path_to_data, 'images', name, img))


if __name__ == '__main__':
    name = 'train'
    path_to_data = '/Users/noam/test/rodent'
    convert_dataloop_to_coco(path_to_data, name)
