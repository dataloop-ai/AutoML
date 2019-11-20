import os
import json
import dtlpy as dl
import tqdm
import traceback
from multiprocessing.pool import ThreadPool
import random
import numpy as np


def create_annotations_txt(annotations_path, images_path,
                           classes_filepath, labels_list, train_split,
                           train_filepath, val_filepath):
    def get_annotations_from_single_file(w_i, w_img_filename):
        try:
            name, ext = os.path.splitext(w_img_filename)
            name = name.replace(images_path, annotations_path)
            annotation_filename = name + '.json'
            img_annotations = list()
            if not os.path.isfile(annotation_filename):
                print('Missing annotations filepath: {}'.format(annotation_filename))
                img_annotations.append('{},,,,,'.format(w_img_filename))
            else:
                with open(annotation_filename, 'r') as f:
                    data = json.load(f)
                current_annotations = dl.AnnotationCollection.from_json(_json=data['annotations'])
                any_annotation_in_file = False
                for annotation in current_annotations.annotations:
                    try:
                        label = annotation.label.lower()
                        if annotation.type != 'box':
                            continue
                        if label in labels_list:
                            top = int(np.round(annotation.top))
                            left = int(np.round(annotation.left))
                            bottom = int(np.round(annotation.bottom))
                            right = int(np.round(annotation.right))
                            if top == bottom:
                                continue
                            if right == left:
                                continue
                            if label not in label_ids:
                                label_id = len(label_ids)
                                label_ids[label] = label_id
                            any_annotation_in_file = True
                            img_annotations.append(
                                '{},{},{},{},{},{}'.format(w_img_filename, left, top, right, bottom, label))
                    except AssertionError:
                        raise
                    except:
                        print(w_img_filename)
                        print(traceback.format_exc())
                        continue
                if not any_annotation_in_file:
                    img_annotations.append('{},,,,,'.format(w_img_filename))
            output_annotations[w_i] = img_annotations
        except:
            print(traceback.format_exc())
        finally:
            pbar.update()

    # get all images from directory
    image_filenames = list()
    for path, su1bdirs, files in os.walk(images_path):
        # break
        for name in files:
            filename, ext = os.path.splitext(name)
            if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
            image_filenames.append(os.path.join(path, name))

    label_ids = dict()
    output_annotations = [[] for _ in range(len(image_filenames))]
    pbar = tqdm.tqdm(total=len(image_filenames))
    pool = ThreadPool(processes=32)
    for i_img_filename, img_filename in enumerate(image_filenames):
        pool.apply_async(get_annotations_from_single_file, kwds={'w_i': i_img_filename,
                                                                 'w_img_filename': img_filename})
    pool.close()
    pool.join()
    pbar.close()

    # write classes
    classes_list = list()
    for key, val in label_ids.items():
        classes_list.append('%s,%s' % (key, val))
    with open(classes_filepath, 'w') as f:
        f.write('\n'.join(classes_list))

    # random shuffle
    random.shuffle(output_annotations)
    n_items = len(output_annotations)
    split_index = int(np.round(train_split * n_items))

    # flat the annotations per item
    train_annotations = [annotation
                         for item_annotations in output_annotations[:split_index]
                         for annotation in item_annotations]
    val_annotations = [annotation
                       for item_annotations in output_annotations[split_index:]
                       for annotation in item_annotations]

    # save training annotations
    with open(train_filepath, 'w') as f:
        f.write('\n'.join(train_annotations))
    # save validation annotations
    with open(val_filepath, 'w') as f:
        f.write('\n'.join(val_annotations))

    print('Num train images: {}'.format(split_index))
    print('Num val images: {}'.format(n_items - split_index))
