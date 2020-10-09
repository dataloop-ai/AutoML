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
from .plotter_lib.BoundingBox import BoundingBox
from .plotter_lib.BoundingBoxes import BoundingBoxes
from .plotter_lib.Evaluator import Evaluator
from .plotter_lib.utils import BBFormat, CoordinatesType, BBType, MethodAveragePrecision
from pycocotools.coco import COCO
import torch
logger = logging.getLogger(__name__)


class precision_recall_compute:

    def __init__(self):
        self.by_model_name = {}

    def _add_image_dets(self, filepathname, model_name):

        bb_type = BBType.Detected

        with open(filepathname, 'r') as f:
            x = f.readlines()
        detections_list = [line.strip('\n').strip(' ').split(' ') for line in x]

        filename = filepathname.strip('/').split('/')[-1]
        image_name = filename.split('.')[0]
        for label, score, x1, y1, x2, y2 in detections_list:
            score, x1, y1, x2, y2 = float(score), float(x1), float(y1), float(x2), float(y2)
            bb = BoundingBox(imageName=image_name,
                             classId=label,
                             x=x1,
                             y=y1,
                             w=x2,
                             h=y2,
                             typeCoordinates=CoordinatesType.Absolute,
                             classConfidence=score,
                             bbType=bb_type,
                             format=BBFormat.XYX2Y2,
                             model_name=model_name)

            if model_name not in self.by_model_name:
                self.by_model_name[model_name] = BoundingBoxes()

            self.by_model_name[model_name].addBoundingBox(bb)

    def add_path_detections(self, predictions_folder, model_name=None):

        for path, subdirs, files in os.walk(predictions_folder):
            if model_name is None:
                if 'check' in path.split('/')[-1]:
                    model_name = path.split('/')[-1]
                else:
                    model_name = 'only_checkpoint'
            for name in files:
                filename, ext = os.path.splitext(name)
                if '.txt' not in ext.lower():
                    continue
                detfilename = os.path.join(path, name)
                self._add_image_dets(detfilename, model_name)

    def add_coco_annotations(self, coco):
        model_name = 'gt'
        bb_type = BBType.GroundTruth

        annotations = coco.dataset['annotations']
        for annotation in annotations:
            x, y, w, h = annotation['bbox']
            category_id = annotation['category_id']
            label = coco.cats[category_id]['name']
            image_id = annotation['image_id']
            image_filename = coco.imgs[image_id]['file_name']
            image_name = image_filename.split('.')[0]
            bb = BoundingBox(imageName=image_name,
                             classId=label,
                             x=x,
                             y=y,
                             w=w,
                             h=h,
                             typeCoordinates=CoordinatesType.Absolute,
                             classConfidence=None,
                             bbType=bb_type,
                             format=BBFormat.XYWH,
                             model_name=model_name)

            if model_name not in self.by_model_name:
                self.by_model_name[model_name] = BoundingBoxes()

            self.by_model_name[model_name].addBoundingBox(bb)

    def _add_dljson(self, filepathname):
        model_name = 'gt'
        bb_type = BBType.GroundTruth
        image_name = filepathname.split('/')[-1].split('.')[0]
        with open(filepathname) as json_file:
            ann_dict = json.load(json_file)

        annotations = ann_dict['annotations']
        for annotation in annotations:
            x1 = annotation['coordinates'][0]['x']
            y1 = annotation['coordinates'][0]['y']
            x2 = annotation['coordinates'][1]['x']
            y2 = annotation['coordinates'][1]['y']
            label = annotation['label']
            bb = BoundingBox(imageName=image_name,
                             classId=label,
                             x=x1,
                             y=y1,
                             w=x2,
                             h=y2,
                             typeCoordinates=CoordinatesType.Absolute,
                             classConfidence=None,
                             bbType=bb_type,
                             format=BBFormat.XYX2Y2,
                             model_name='groundtruth')

            if model_name not in self.by_model_name:
                self.by_model_name[model_name] = BoundingBoxes()

            self.by_model_name[model_name].addBoundingBox(bb)

    def add_dataloop_local_annotations(self, json_file):
        for path, subdirs, files in os.walk(json_file):
            for name in files:
                filename, ext = os.path.splitext(name)
                if '.json' not in ext.lower():
                    continue
                gt_filename = os.path.join(path, name)
                self._add_dljson(gt_filename)

    def _collect_annotations(self, w_item):
        try:
            # add all box annotations to calculation
            # print(w_item.filename)
            annotations = w_item.annotations.list()
            thisItem = BoundingBoxes()
            for annotation in annotations:
                if annotation.type != 'box':
                    continue
                try:
                    model_name = annotation.metadata['user']['model']['name']
                    bb_type = BBType.Detected
                    class_conf = annotation.metadata['user']['model']['confidence']
                except KeyError:
                    bb_type = BBType.GroundTruth
                    model_name = 'gt'
                    class_conf = None
                except:
                    raise
                if model_name not in self.models_alowed_ls:
                    continue
                image_name = w_item.filename.split('/')[-1].split('.')[0]
                bb = BoundingBox(imageName=image_name,
                                 classId=annotation.label,
                                 x=annotation.coordinates[0]['x'],
                                 y=annotation.coordinates[0]['y'],
                                 w=annotation.coordinates[1]['x'],
                                 h=annotation.coordinates[1]['y'],
                                 typeCoordinates=CoordinatesType.Absolute,
                                 classConfidence=class_conf,
                                 bbType=bb_type,
                                 format=BBFormat.XYX2Y2,
                                 model_name=model_name)
                thisItem.addBoundingBox(bb)
                lock = threading.Lock()
                with lock:
                    if model_name not in self.by_model_name:
                        self.by_model_name[model_name] = BoundingBoxes()

                    self.by_model_name[model_name].addBoundingBox(bb)
        except:
            print(traceback.format_exc())

    def add_dataloop_remote_annotations(self, project_name, dataset_name, filter_value, model_name):
        dlp.setenv('prod')
        project = dlp.projects.get(project_name)
        dataset = project.datasets.get(dataset_name)

        filters = dlp.Filters()
        filters.add(field='filename', values=filter_value)
        pages = dataset.items.list(filters=filters)
        self.models_alowed_ls = [model_name, 'gt']
        i_item = 0
        pool = ThreadPool(processes=32)
        for page in pages:
            for item in page:
                if item.filename.startswith('/.dataloop'):
                    continue
                pool.apply_async(self._collect_annotations, kwds={'w_item': item})
                i_item += 1
        pool.close()
        pool.join()

    def plot_metrics(self):
        precision_recall_fig, precision_recall = plt.subplots()
        fig = plt.figure()
        scores_recall = fig.add_subplot(1, 1, 1)
        fig = plt.figure()
        precision_scores = fig.add_subplot(1, 1, 1)
        evaluator = Evaluator()
        model_names = list()
        legend = list()
        class_id = 0
        lines = list()
        for model_name, bbs in self.by_model_name.items():
            model_names.append(model_name)
            if model_name in ['gt']:
                continue

            to_show = BoundingBoxes()
            to_show._boundingBoxes += bbs._boundingBoxes
            to_show._boundingBoxes += self.by_model_name['gt']._boundingBoxes
            res = evaluator.GetPascalVOCMetrics(boundingboxes=to_show,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation,
                                                confidence_threshold=0.)
            results = res[class_id]
            # plot precision recall
            lines.append(precision_recall.plot(results['cum_recalls'], results['cum_precisions'], label=model_name)[0])
            # plot scores
            scores_recall.plot(results['cum_recalls'], results['scores'], label=model_name)
            # plot scores
            precision_scores.plot(results['scores'], results['cum_precisions'], label=model_name)
            legend.append(model_name)
        # final
        p_min = -0.05
        p_max = 1.05
        for fig in [precision_recall, scores_recall, precision_scores]:
            fig.set_xlim(p_min, p_max)
            fig.set_ylim(p_min, p_max)
            major_ticks = np.linspace(0, 1, 11)
            minor_ticks = np.linspace(0, 1, 21)
            fig.set_xticks(major_ticks)
            fig.set_xticks(minor_ticks, minor=True)
            fig.set_yticks(major_ticks)
            fig.set_yticks(minor_ticks, minor=True)
            # And a corresponding grid
            fig.grid(which='both')
            # Or if you want different settings for the grids:
            fig.grid(which='minor', alpha=0.4)
            fig.grid(which='major', alpha=0.8)
            fig.legend(legend)

        precision_recall.set_ylabel('precision')
        precision_recall.set_xlabel('recall')
        precision_recall.set_title('Precision Recall')

        scores_recall.set_ylabel('scores')
        scores_recall.set_xlabel('recall')
        scores_recall.set_title('Score Recall')

        precision_scores.set_ylabel('precision')
        precision_scores.set_xlabel('scores')
        precision_scores.set_title('Precision Score')
        ##############
        leg = precision_recall.legend(loc='lower left', fancybox=True, shadow=True)
        leg.get_frame().set_alpha(0.4)

        # we will set up a dict mapping legend line to orig line, and enable
        # picking on the legend line
        lined = dict()
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)  # 5 pts tolerance
            lined[legline] = origline

        def onpick(event):
            # on the pick event, find the orig line corresponding to the
            # legend proxy line, and toggle the visibility
            legline = event.artist
            origline = lined[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            precision_recall_fig.canvas.draw()

        precision_recall_fig.canvas.mpl_connect('pick_event', onpick)

        plt.show()

    def save_plot_metrics(self):
        precision_recall_fig, precision_recall = plt.subplots()

        evaluator = Evaluator()
        model_names = list()
        legend = list()
        class_id = 0
        lines = list()
        for model_name, bbs in self.by_model_name.items():
            model_names.append(model_name)
            if model_name in ['gt']:
                continue

            to_show = BoundingBoxes()
            to_show._boundingBoxes += bbs._boundingBoxes
            to_show._boundingBoxes += self.by_model_name['gt']._boundingBoxes
            res = evaluator.GetPascalVOCMetrics(boundingboxes=to_show,
                                                IOUThreshold=0.5,
                                                method=MethodAveragePrecision.EveryPointInterpolation,
                                                confidence_threshold=0.)
            results = res[class_id]
            # plot precision recall
            lines.append(precision_recall.plot(results['cum_recalls'], results['cum_precisions'], label=model_name)[0])

            legend.append(model_name)
        # final
        p_min = -0.05
        p_max = 1.05
        for fig in [precision_recall]:
            fig.set_xlim(p_min, p_max)
            fig.set_ylim(p_min, p_max)
            major_ticks = np.linspace(0, 1, 11)
            minor_ticks = np.linspace(0, 1, 21)
            fig.set_xticks(major_ticks)
            fig.set_xticks(minor_ticks, minor=True)
            fig.set_yticks(major_ticks)
            fig.set_yticks(minor_ticks, minor=True)
            # And a corresponding grid
            fig.grid(which='both')
            # Or if you want different settings for the grids:
            fig.grid(which='minor', alpha=0.4)
            fig.grid(which='major', alpha=0.8)
            fig.legend(legend)

        precision_recall.set_ylabel('precision')
        precision_recall.set_xlabel('recall')
        precision_recall.set_title('Precision Recall')

        ##############
        leg = precision_recall.legend(loc='lower left', fancybox=True, shadow=True)
        leg.get_frame().set_alpha(0.4)

        # we will set up a dict mapping legend line to orig line, and enable
        # picking on the legend line
        lined = dict()
        for legline, origline in zip(leg.get_lines(), lines):
            legline.set_picker(5)  # 5 pts tolerance
            lined[legline] = origline

        def onpick(event):
            # on the pick event, find the orig line corresponding to the
            # legend proxy line, and toggle the visibility
            legline = event.artist
            origline = lined[legline]
            vis = not origline.get_visible()
            origline.set_visible(vis)
            # Change the alpha on the line in the legend so we can see what lines
            # have been toggled
            if vis:
                legline.set_alpha(1.0)
            else:
                legline.set_alpha(0.2)
            precision_recall_fig.canvas.draw()

        precision_recall_fig.canvas.mpl_connect('pick_event', onpick)

        plt.savefig('precision_recall.png')

    def get_metric(self, model_name, precision_to_recall_ratio=1.):
        evaluator = Evaluator()
        model_names = list()
        class_id = 0
        logger.info
        bbs = self.by_model_name[model_name]

        to_show = BoundingBoxes()
        to_show._boundingBoxes += bbs._boundingBoxes
        to_show._boundingBoxes += self.by_model_name['gt']._boundingBoxes
        res = evaluator.GetPascalVOCMetrics(boundingboxes=to_show,
                                            IOUThreshold=0.5,
                                            method=MethodAveragePrecision.EveryPointInterpolation,
                                            confidence_threshold=0.,
                                            precision_to_recall_ratio=precision_to_recall_ratio)
        sum_AP = 0.
        total_num_annotations = 0
        for r in res:
            sum_AP += r['AP'] * r['num_annotations']
            total_num_annotations += r['num_annotations']

        if total_num_annotations > 0:
            mAP = sum_AP / total_num_annotations
        else:
            mAP = 0
        # plot precision recall
        return mAP




if __name__ == '__main__':
    predictions_dir_path = '/Users/noam/data/rodent_data/predictions'

    gt_file = glob.glob(os.path.join(predictions_dir_path, '*.json'))[0]
    json_file_path = os.path.join(predictions_dir_path, 'json')
    coco_object = COCO(gt_file)

    plotter = precision_recall_compute()

    # compute.add_coco_annotations(coco_object)
    # compute.add_dataloop_local_annotations(json_file_path)
    plotter.add_path_detections(predictions_dir_path)

    plotter.add_dataloop_remote_annotations(project_name='IPM SQUARE EYE', dataset_name='Rodents',
                                             filter_value='/Pics_from_Mice_EYE/TestSet/**',
                                             model_name='retinanet_resnet101_custom_anchors_02_2020')

    new_checkpoint_mAP = plotter.get_metric(model_name='new_checkpoint', precision_to_recall_ratio=1.)
    plotter.save_plot_metrics()
