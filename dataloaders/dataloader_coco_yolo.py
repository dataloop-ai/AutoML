import glob
import os
import cv2
import numpy
import skimage
import skimage.color
import skimage.io
import skimage.transform
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from custom_transforms import *
from utils import draw_bbox
from importlib import import_module


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

        if self.built_in_transforms is not None:
            self.into_module()


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
            img = cv2.imread(self.img_path_list[index])
            dh, dw, _ = img.shape
            ann = []
            fl = open(self.ann_path_list[index], 'r')
            for dt in fl.readlines():  # 依次读取每行
                dt = dt.strip()  # 去掉每行头尾空白
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
            ann=numpy.asarray(ann)
            sample = {'img': img, 'annot': ann}
            if self.function_transforms is not None:
                for tsfm in self.function_transforms:

                    sample = tsfm(sample)

            return sample

        elif self.data_format == 'coco':
            img = self.load_image(index)
            annot = self.load_annotations(index)

            sample = {'img': img, 'annot': annot}


            if self.function_transforms is not None:
                for tsfm in self.function_transforms:
                    sample = tsfm(sample)

            return sample

    def __len__(self):
        if self.data_format == 'yolo':
            return len(self.img_path_list)
        elif self.data_format == 'coco':
            return len(self.image_ids)

    def into_module(self):
        for module in self.built_in_transforms:
            m1 = getattr(import_module(module), module)
            m = m1()


    def visualize(self,save_path):
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
                        # img_normoalize = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                        # image_pil = Image.fromarray(img_normoalize)
                        # img_cv2 = cv2.cvtColor(numpy.asarray(image_pil), cv2.COLOR_RGB2BGR)

                        for bbox in annot:
                            label = bbox[4]
                            x1 = int(bbox[0])
                            y1 = int(bbox[1])
                            x2 = int(bbox[2])
                            y2 = int(bbox[3])
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    cv2.imwrite(save_img_path, img)



            else:
                for img_path, ann_path in zip(self.img_path_list, self.ann_path_list):
                    img = cv2.imread(img_path)
                    filename = img_path.split("/")[-1]

                    dh, dw, _ = img.shape
                    fl = open(ann_path, 'r')
                    data = fl.readlines()
                    fl.close()
                    # Taken from https://stackoverflow.com/questions/64096953/how-to-convert-yolo-format-bounding-box-coordinates-into-opencv-format
                    for dt in data:
                        # Split string to float
                        _, x, y, w, h = map(float, dt.split(' '))
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

                        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)
                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    cv2.imwrite(save_img_path, img)

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
                        img_normoalize = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
                        image_pil = Image.fromarray(img_normoalize)
                        img_cv2 = cv2.cvtColor(numpy.asarray(image_pil), cv2.COLOR_RGB2BGR)

                        for bbox in annot:
                            label = bbox[4]
                            x1 = int(bbox[0])
                            y1 = int(bbox[1])
                            x2 = int(bbox[2])
                            y2 = int(bbox[3])
                            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), (0, 255, 0),1)

                    os.makedirs(save_path, exist_ok=True)
                    save_img_path = os.path.join(save_path, filename)
                    cv2.imwrite(save_img_path, img_cv2)
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

    def num_classes(self):
        if self.data_format == 'yolo':
            return len(self.classes_set)
        elif self.data_format == 'coco':
            return len(self.classes)



def test1():
    # yolo with function_transforms
    y = CustomDataset('/Users/yi-chu/Downloads/mask_dataset','yolo')
    # y.visualize('/Users/yi-chu/Downloads/yoloVisualize1')
    print("before: ", y[3])
    print("items of y[3]: ", y[3].items())
    print("key of y[3]: ",y[3].keys())
    print("shape of y[3]['img']: ", y[3]['img'].shape)
    print("shape of y[3]['annot']: ", y[3]['annot'].shape)
    print("type of y[3]['img']: ", type(y[3]['img']))
    print("type of y[3]['annot']: ", type(y[3]['annot']))

    y = CustomDataset('/Users/yi-chu/Downloads/mask_dataset', 'yolo', function_transforms=[CutOut(10), ])
    print("after: ", y[3])
    print("items of y[3]: ", y[3].items())
    print("key of y[3]: ",y[3].keys())
    print("shape of y[3]['img']: ", y[3]['img'].shape)
    print("shape of y[3]['annot']: ", y[3]['annot'].shape)
    print("type of y[3]['img']: ", type(y[3]['img']))
    print("type of y[3]['annot']: ", type(y[3]['annot']))


    # y = CustomDataset('/Users/yi-chu/Downloads/mask_dataset','yolo',function_transforms=[CutOut(10), ])
    # y.visualize('/Users/yi-chu/Downloads/yoloVisualize2')
    # print("after: ",y[3])

def test2():
    # coco with function_transforms

    # c.visualize('/Users/yi-chu/Downloads/cocoVisualize1')
    # print("before: ", c[3])
    #
    # c.visualize('/Users/yi-chu/Downloads/cocoVisualize2')
    # print("after: ", c[3])


    y = CustomDataset('/Users/yi-chu/Downloads/archive/train', 'coco' )
    print("before: ", y[3])
    print("items of y[3]: ", y[3].items())
    print("key of y[3]: ",y[3].keys())
    print("shape of y[3]['img']: ", y[3]['img'].shape)
    print("shape of y[3]['annot']: ", y[3]['annot'].shape)
    print("type of y[3]['img']: ", type(y[3]['img']))
    print("type of y[3]['annot']: ", type(y[3]['annot']))

    y = CustomDataset('/Users/yi-chu/Downloads/archive/train', 'coco', function_transforms=[CutOut(10), ])
    print("after: ", y[3])
    print("items of y[3]: ", y[3].items())
    print("key of y[3]: ",y[3].keys())
    print("shape of y[3]['img']: ", y[3]['img'].shape)
    print("shape of y[3]['annot']: ", y[3]['annot'].shape)
    print("type of y[3]['img']: ", type(y[3]['img']))
    print("type of y[3]['annot']: ", type(y[3]['annot']))








def test3():
    #generator
    c = CustomDataset('/Users/yi-chu/Downloads/archive/train', 'coco')
    dataloader_iterator = iter(c)
    data = next(dataloader_iterator)
    print(data)
    data = next(dataloader_iterator)
    print(data)
    data = next(dataloader_iterator)
    print(data)


if __name__=='__main__':

    # c = CustomDataset('/Users/yi-chu/Downloads/archive/train', 'coco',built_in_transforms=['hello'])
   test2()