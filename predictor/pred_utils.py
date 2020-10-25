import numpy as np
import time
import os
import skimage
import skimage.io
import skimage.color
import cv2
import torch
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms
from copy import deepcopy

from networks.retinanet import ret50
from dataloaders import PredDataset, collater, Resizer, Normalizer, UnNormalizer

try:
    from logging_utils import logginger
    logger = logginger(__name__)
except:
    import logging
    logger = logging.getLogger(__name__)


def detect(checkpoint, pred_on_path, output_path, threshold=0.5, visualize=False, red_label='sick'):
    device = torch.device(type='cuda') if torch.cuda.is_available() else torch.device(type='cpu')

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
        os.makedirs(output_path)
    logger.info('inside ' + str(pred_on_path) + ': ' + str(os.listdir(pred_on_path)))
    dataset_val = PredDataset(pred_on_path=pred_on_path,
                              transform=transforms.Compose([Normalizer(), Resizer(min_side=608)])) #TODO make resize an input param
    logger.info('dataset prepared')
    dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=None)
    logger.info('data loader initialized')
    labels = checkpoint['labels']
    logger.info('labels are: ' + str(labels))
    num_classes = len(labels)
    configs = deepcopy(checkpoint['model_specs']['training_configs'])
    configs.update(checkpoint['hp_values'])
    logger.info('initializing object_detection model')
    retinanet = ret50(num_classes=num_classes, scales=configs['anchor_scales'], ratios=configs['anchor_ratios']) #TODO: make depth an input parameter
    logger.info('loading weights')
    retinanet.load_state_dict(checkpoint['model'])
    retinanet = retinanet.to(device=device)
    logger.info('model to device: ' + str(device))
    retinanet.eval()
    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    inference_times = []
    for idx, data in enumerate(dataloader_val):
        scale = data['scale'][0]
        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].to(device=device).float())
            elapsed_time = time.time() - st
            print('Elapsed time: {}'.format(elapsed_time))
            inference_times.append(elapsed_time)
            idxs = np.where(scores.cpu() > threshold)[0]
            if visualize:
                img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

                img[img < 0] = 0
                img[img > 255] = 255

                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            detections_list = []
            for j in range(idxs.shape[0]):
                bbox = transformed_anchors[idxs[j], :]
                if visualize:
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])

                label_idx = int(classification[idxs[j]])
                label_name = labels[label_idx]
                score = scores[idxs[j]].item()
                if visualize:
                    draw_caption(img, (x1, y1, x2, y2), label_name)
                    if red_label in label_name:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    print(label_name)

                # un resize for eval against gt
                bbox /= scale
                bbox.round()
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                detections_list.append([label_name, str(score), str(x1), str(y1), str(x2), str(y2)])
            img_name = dataset_val.image_names[idx].split('/')[-1]
            i_name = img_name.split('.')[0]
            filename = i_name + '.txt'
            filepathname = os.path.join(output_path, filename)
            with open(filepathname, 'w', encoding='utf8') as f:
                for single_det_list in detections_list:
                    for i, x in enumerate(single_det_list):
                        f.write(str(x))
                        f.write(' ')
                    f.write('\n')
            if visualize:
                save_to_path = os.path.join(output_path, img_name)
                cv2.imwrite(save_to_path, img)
                cv2.waitKey(0)
    print('average inference time per image: ', np.mean(inference_times))
    return output_path

def detect_single_image(checkpoint, image_path, visualize=False):
    device = torch.device(type='cuda') if torch.cuda.is_available() else torch.device(type='cpu')
    configs = deepcopy(checkpoint['model_specs']['training_configs'])
    configs = configs.update(checkpoint['hp_values'])
    labels = checkpoint['labels']
    num_classes = len(labels)
    retinanet = ret50(num_classes=num_classes, scales=configs['anchor_scales'], ratios=configs['anchor_ratios']) #TODO: make depth an input parameter
    retinanet.load_state_dict(checkpoint['model'])
    retinanet = retinanet.to(device=device)
    retinanet.eval()

    img = skimage.io.imread(image_path)

    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    img = img.astype(np.float32) / 255.0
    transform = transforms.Compose([Normalizer(), Resizer(min_side=608)]) #TODO: make this dynamic
    data = transform({'img': img, 'annot': np.zeros((0, 5))})
    img = data['img']
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    with torch.no_grad():
        scores, classification, transformed_anchors = retinanet(img.to(device=device).float())


        idxs = np.where(scores.cpu() > 0.5)[0]
        scale = data['scale']
        detections_list = []
        for j in range(idxs.shape[0]):
            bbox = transformed_anchors[idxs[j], :]
            label_idx = int(classification[idxs[j]])
            label_name = labels[label_idx]
            score = scores[idxs[j]].item()

            # un resize for eval against gt
            bbox /= scale
            bbox.round()
            x1 = int(bbox[0])
            y1 = int(bbox[1])
            x2 = int(bbox[2])
            y2 = int(bbox[3])
            detections_list.append([label_name, str(score), str(x1), str(y1), str(x2), str(y2)])
        img_name = image_path.split('/')[-1].split('.')[0]
        filename = img_name + '.txt'
        path = os.path.dirname(image_path)
        filepathname = os.path.join(path, filename)
        with open(filepathname, 'w', encoding='utf8') as f:
            for single_det_list in detections_list:
                for i, x in enumerate(single_det_list):
                    f.write(str(x))
                    f.write(' ')
                f.write('\n')

        if visualize:
            unnormalize = UnNormalizer()


    return filepathname

