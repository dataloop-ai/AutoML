import matplotlib.pyplot as plt
import numpy as np
import traceback
from multiprocessing.pool import ThreadPool
import json
import glob
import os
import threading
import logging
import dtlpy as dlp
from pycocotools.coco import COCO
from plotmetriclib.plot_curve import precision_recall_compute
import torch
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    predictions_dir_path = '/Users/noam/data/rodent_data/predictions'

    gt_file = glob.glob(os.path.join(predictions_dir_path, '*.json'))[0]
    json_file_path = os.path.join(predictions_dir_path, 'json')
    coco_object = COCO(gt_file)

    plotter = precision_recall_compute()

    # compute.add_coco_annotations(coco_object)
    # compute.add_dataloop_local_annotations(json_file_path)
    plotter.add_path_detections(predictions_dir_path, model_name='new_checkpoint')

    plotter.add_dataloop_remote_annotations(project_name='IPM SQUARE EYE', dataset_name='Rodents',
                                             filter_value='/Pics_from_Mice_EYE/TestSet/**',
                                             model_name='retinanet_resnet101_custom_anchors_02_2020')

    new_checkpoint_mAP = plotter.get_metric(model_name='new_checkpoint', precision_to_recall_ratio=1.)
    plotter.save_plot_metrics()