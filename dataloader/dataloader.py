import csv
import glob
import os
import sys

import cv2
import numpy
import skimage
import skimage.color
import skimage.io
import skimage.transform
from pycocotools.coco import COCO
from torch.utils.data import Dataset, Sampler
from .custom_transforms import *
from .utils import draw_bbox
from importlib import import_module
from .image import ImageData


class CustomDataset(Dataset):
    def __init__(self, img_path, data_format, function_transforms=None, built_in_transforms=None):
        """
        Args:
            img_path: the dataset path
            data_format: dataset format. i.e. CoCo or Yolo.
            functions_of_transform" is a list of string with single/variety of transform functions.
            built_in_augmentations: is a list of string with single/variety of augmentations in the library.

        """

        self.img_path = img_path
        self.data_format = data_format
        self.function_transforms = function_transforms
        self.built_in_transforms = built_in_transforms


        self.ann_path_list = []
        if self.data_format == 'yolo':
            # get image list
            self.img_path_list = glob.glob(img_path + '/' + '*.jpg')
            # get annotation list
            self.ann_path_list = glob.glob(img_path + '/'+'*.txt')
            self.classes_set = set()
            self.calculate_classes()
            self.img_path_list.sort()
            self.ann_path_list.sort()

        elif self.data_format == 'coco':
            self.set_name = 'train'
            self.img_path_list = glob.glob(img_path + '/images/' + self.set_name + '/'+'*.jpg')
            self.coco = COCO(os.path.join(self.img_path, 'annotations', 'instances_' + self.set_name + '.json'))
            self.image_ids = self.coco.getImgIds()
            self.img_path_list.sort()

            self.load_classes()
        # if built_in_transforms is not None:
        #     for b in self.built_in_transforms:
        #         importlib.import_module(b)

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __getitem__(self, index):




        if self.data_format == 'yolo':
            img = self.load_image_yolo(index)
            dh, dw, _ = img.shape
            annot = self.load_annotations_yolo(index, dh, dw)

            sample = {'img': img, 'annot': annot}

            if self.function_transforms is not None:
                for tsfm in self.function_transforms:
                    sample = tsfm(sample)

            elif self.built_in_transforms is not None:
                sample = self.add_module(sample)

            if 'scale' in sample:
                image_data = ImageData(sample['img'], sample['annot'], sample['scale'])
            else:
                image_data = ImageData(sample['img'], sample['annot'])
            return image_data

        elif self.data_format == 'coco':
            img = self.load_image(index)
            annot = self.load_annotations(index)

            sample = {'img': img, 'annot': annot}
        

            if self.function_transforms is not None:
                for tsfm in [self.function_transforms]:
                    sample = tsfm(sample)
            elif self.built_in_transforms is not None:
                sample = self.add_module(sample)

            if 'scale' in sample:
                image_data = ImageData(sample['img'], sample['annot'], sample['scale'])
            else:
                image_data = ImageData(sample['img'], sample['annot'])
            return image_data

    def __len__(self):
        if self.data_format == 'yolo':
            return len(self.img_path_list)
        elif self.data_format == 'coco':
            return len(self.image_ids)

    def add_module(self, sample):
        for module in self.built_in_transforms:
            m1 = getattr(import_module('dataloader'), module)
            result = m1()(sample)
        return result

    def visualize(self, save_path):
        if self.data_format == 'yolo':
            sample_list = []
            file = []
            if self.function_transforms is not None:
                for image in range(len(self.img_path_list)):
                    filename = self.img_path_list[image].split("/")[-1]
                    sample_list.append(self.__getitem__(image))
                    file.append(filename)

                    for sample in sample_list:

                        img = sample['img']
                        annot = sample['annot']

                        for bbox in annot:
                            label = bbox[4]
                            x1 = int(bbox[0])
                            y1 = int(bbox[1])
                            x2 = int(bbox[2])
                            y2 = int(bbox[3])
                            bbox = [x1, y1, x2, y2]
                            draw_bbox(img, bbox, label)


                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    skimage.io.imsave(save_img_path, img)

            else:
                for img_path, ann_path in zip(self.img_path_list, self.ann_path_list):
                    img = skimage.io.imread(img_path)
                    filename = img_path.split("/")[-1]

                    dh, dw, _ = img.shape
                    fl = open(ann_path, 'r')
                    data = fl.readlines()
                    fl.close()
                    # Taken from https://stackoverflow.com/questions/64096953/how-to-convert-yolo-format-bounding-box-coordinates-into-opencv-format
                    for dt in data:
                        # Split string to float
                        c, x, y, w, h = map(float, dt.split(' '))
                        left = int((x - w / 2) * dw)
                        right = int((x + w / 2) * dw)
                        top = int((y - h / 2) * dh)
                        bottom = int((y + h / 2) * dh)
                        if left < 0:
                            left = 0
                        if right > dw - 1:
                            right = dw - 1
                        if top < 0:
                            top = 0
                        if bottom > dh - 1:
                            bottom = dh - 1
                        bbox = [left, top, right, bottom]
                        draw_bbox(img, bbox, c)


                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    skimage.io.imsave(save_img_path, img)

        elif self.data_format == 'coco':
            if self.function_transforms is not None:
                sample_list = []
                file = []
                for image in range(len(self.img_path_list)):
                    filename = self.img_path_list[image].split("/")[-1]
                    sample_list.append(self.__getitem__(image))
                    file.append(filename)

                    for sample in sample_list:
                        img = sample['img']
                        annot = sample['annot']

                        for bbox in annot:
                            label = bbox[4]
                            draw_bbox(img, bbox[:4], label)

                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    skimage.io.imsave(save_img_path, img)
            else:
                for idx in range(len(self.image_ids)):
                    img = self.load_image(idx)
                    annot = self.load_annotations(idx)
                    for bbox in annot:
                        label = self.labels[bbox[4]]
                        draw_bbox(img, bbox[:4], label)
                    filename = self.coco.loadImgs(self.image_ids[idx])[0]['file_name']
                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    skimage.io.imsave(save_img_path, img)

    def load_image_yolo(self,image_index):
        path = self.img_path_list[image_index]
        try:
            img = skimage.io.imread(path)
            if len(img.shape) == 2:
                img = skimage.color.gray2rgb(img)
            return img.astype(np.float32) / 255.0

        except Exception as e:
            print(e)

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.img_path, 'images', self.set_name, image_info['file_name'])
        try:
            img = skimage.io.imread(path)
            if len(img.shape) == 2:
                img = skimage.color.gray2rgb(img)
            return img.astype(np.float32) / 255.0

        except Exception as e:

            raise Exception('image name: ' + image_info['file_name'] + ', id: ' + str(image_info[
                'id']) + ' caused the following error ' + repr(e))

    def load_annotations_yolo(self, index, dh, dw):

        ann = []
        fl = open(self.ann_path_list[index], 'r')
        for dt in fl.readlines():
            dt = dt.strip()
            # Split string to float
            c, x, y, w, h = map(float, dt.split(' '))
            left = int((x - w / 2) * dw)
            right = int((x + w / 2) * dw)
            top = int((y - h / 2) * dh)
            bottom = int((y + h / 2) * dh)
            if left < 0:
                left = 0
            if right > dw - 1:
                right = dw - 1
            if top < 0:
                top = 0
            if bottom > dh - 1:
                bottom = dh - 1

            temp_ann = [left, top, right, bottom, c]
            ann.append(temp_ann)

        fl.close()
        return numpy.asarray(ann)

    def load_annotations(self, image_index):
        # get ground truth annotations in [x1, y1, x2, y2] format
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    # These two functions are so the network has every label from 0 - 80 consistently
    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def calculate_classes(self):
        for ann_file in self.ann_path_list:
            with open(ann_file) as f:
                ann = f.readline().split(" ")
                self.classes_set.add(ann[0])
    @property
    def num_classes(self):
        if self.data_format == 'yolo':
            return len(self.classes_set)
        elif self.data_format == 'coco':
            return len(self.classes)


class PredDataset(Dataset):
    """CSV dataset."""

    def __init__(self, pred_on_path, class_list_path=None, transform=None, resize=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = pred_on_path
        self.class_list_path = class_list_path
        self.transform = transform

        # parse the provided class file
        if self.class_list_path is not None:
            try:
                with self._open_for_csv(self.class_list_path) as file:
                    self.classes = self.load_classes(csv.reader(file, delimiter=','))
            except ValueError as e:
                raise (ValueError('invalid CSV class file: {}: {}'.format(self.class_list_path, e)), None)

            self.labels = {}
            for key, value in self.classes.items():
                self.labels[value] = key
        full_names = []
        for name in os.listdir(pred_on_path):
            try:
                if name.split('.')[1] in ['jpg', 'png']:
                    full_names.append(os.path.join(pred_on_path, name))
            except:
                pass
        image_data = {}
        for full_name in full_names:
            image_data[full_name] = []

        self.image_data = image_data
        self.image_names = full_names
        self.resize = resize
        self.transform_this = self.get_transform()

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise (ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):

            try:
                class_name, class_id = row
            except ValueError:
                class_name = row[0]
                class_id = line
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
            line += 1
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
            # sample = self.transform_this(sample)

        return sample

    def get_transform(self):
        return TransformTr(resize=self.resize)

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a['class'])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            img_file = row[0]
            if img_file not in result:
                result[img_file] = []

        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    
    def num_classes(self):
        return len(self.classes)

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None, resize=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise (ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise (ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)), None)
        self.image_names = list(self.image_data.keys())
        self.resize = resize
        self.transform_this = self.get_transform()

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise (ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):

            try:
                class_name, class_id = row
            except ValueError:
                class_name = row[0]
                class_id = line
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
            line += 1
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
            # sample = self.transform_this(sample)

        return sample

    def get_transform(self):
        return TransformTr(resize=self.resize)

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a['class'])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise (ValueError(
                    'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                       None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return len(self.classes)

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):
    imgs = [s.image for s in data]
    annots = [s.annot for s in data]
    scales = [s.scale for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:  

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    image_data = ImageData(padded_imgs, annot_padded,scales)

    return image_data

def detection_augment_list():
    l = [
        (Translate_Y, -0.3, 0.3),
        (Translate_Y_BBoxes, -0.3, 0.3),
        (Translate_X, -0.3, 0.3),
        (Translate_X_BBoxes, -0.3, 0.3),
        (CutOut, 6, 20),
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


class Augmentation(object):
    def __init__(self, policies, detection=True):
        self.policies = policies
        self.detection = detection

    def __call__(self, sample):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                sample = apply_augment(sample, name, level, self.detection)
        return sample


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, min_side=608, max_side=1024):
        self.min_side = min_side
        self.max_side = max_side

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape
        # add padding for fpn
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}




class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]






def test1():
    # yolo with function_transforms
    y = CustomDataset('/Users/yi-chu/Downloads/mask_dataset','yolo')
    print("before: ", y[3])
    print("image : ", y[3].img.shape)
    print("image type:", type(y[3].img))

    print("image : ", y[3].annot.shape)
    print("image type:", type(y[3].annot))
    y[3].visualize()


    y = CustomDataset('/Users/yi-chu/Downloads/mask_dataset', 'yolo', function_transforms=[CutOut(10), ])
    print("after: ", y[3])

    print("image shape: ", y[3].img.shape)
    print("annotation type:", type(y[3].img))

    print("image shape: ", y[3].annot.shape)
    print("annotation type:", type(y[3].annot))
    y[3].visualize()


def test2():

    y = CustomDataset('/root/data/car_dataset/train', 'coco' )
    print("before: ", y[3])
    print("image shape: ", y[3].img.shape)
    print("annotation type:", type(y[3].img))

    print("image : ", y[3].annot.shape)
    print("image type:", type(y[3].annot))


    y = CustomDataset('/root/data/car_dataset/train', 'coco', function_transforms=[CutOut(10), ])
    print("after: ", y[3])

    print("image shape: ", y[3].img.shape)
    print("annotation type:", type(y[3].img))

    print("image shape: ", y[3].annot.shape)
    print("annotation type:", type(y[3].annot))


def test3():

    #visualize
    c = CustomDataset('/root/data/car_dataset/train', 'coco')
    c[6].visualize('/root/dataloader/')
    c = CustomDataset('/root/data/car_dataset/train', 'coco', function_transforms=[CutOut(10), ])
    c[6].visualize()

def test4():

    # buil in transform  before and after
    y = CustomDataset('/root/data/mask_dataset', 'yolo')
    y[4].visualize()
    y = CustomDataset('/root/data/mask_dataset', 'yolo',built_in_transforms=['Resizer'])
    y[4].visualize()
    print(y[4])

if __name__=='__main__':


    test3()
    test4()

    test1()
    test2()




