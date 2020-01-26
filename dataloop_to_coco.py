from multiprocessing.pool import ThreadPool
import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import threading
import traceback
import dtlpy as dl
import pylab


def collect_annotations(w_item):
    try:
        # add all box annotations to calculation
        # print(w_item.filename)
        annotations = w_item.annotations.list()
        for annotation in annotations:
            if annotation.type != 'box':
                continue
            try:
                model_name = annotation.metadata['user']['model']['name']
                bb_type = 'model'
                class_conf = annotation.metadata['user']['model']['confidence']
            except KeyError:
                bb_type = 'gt'
                model_name = 'gt'
                class_conf = None
            except:
                raise

            for img in images:
                if w_item.name in img['file_name']:
                    img_id = img['id']
                    break
            x = annotation.coordinates[0]['x']
            y = annotation.coordinates[0]['y']
            w = annotation.coordinates[1]['x'] - x
            h = annotation.coordinates[1]['y'] - y

            ann = {'bbox': [x, y, w, h],
                   'area': w * h,
                   'iscrowd': 0,
                   'category_id': label_to_id[annotation.label],
                   'image_id': img_id,
                   'score': annotation.attributes[0] if annotation.attributes else 1}
            with lock:
                if bb_type == 'gt':
                    ann['id'] = len(cocoGt.dataset['annotations']) + 1
                    cocoGt.dataset['annotations'].append(ann)
                else:
                    ann['id'] = len(cocoDt.dataset['annotations']) + 1
                    cocoDt.dataset['annotations'].append(ann)
    except:
        print(traceback.format_exc())


lock = threading.Lock()
# initialize COCO ground truth api
with open('../mice_data/Rodent.txt', 'r') as f:
    class_list = f.read().split('\n')

label_to_id = {name: i for i, name in enumerate(class_list)}
images = [{'file_name': filename,
           'id': i}
          for i, filename in enumerate(os.listdir(r'E:\Datasets\COCO\2017\images\val2017'))]

cocoGt = COCO()
cocoDt = COCO()
cocoGt.dataset['categories'] = [{'id': i, 'name': name} for name, i in label_to_id.items()]
cocoDt.dataset['categories'] = [{'id': i, 'name': name} for name, i in label_to_id.items()]
cocoGt.dataset['annotations'] = list()
cocoDt.dataset['annotations'] = list()
cocoGt.dataset['images'] = images
cocoDt.dataset['images'] = images

dl.setenv('prod')
project = dl.projects.get('COCO')
dataset = project.datasets.get('val')
pages = dataset.items.list()
pool = ThreadPool(processes=32)
for item in pages.all():
    pool.apply_async(collect_annotations, kwds={'w_item': item})
pool.close()
pool.join()
cocoGt.createIndex()
cocoDt.createIndex()

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm', 'bbox', 'keypoints']
annType = annType[1]  # specify type here

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, annType)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()