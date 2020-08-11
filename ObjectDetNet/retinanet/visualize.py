import numpy as np
import time
import glob
import os
import cv2
from . import model
import torch
from shutil import copyfile
from torch.utils.data import DataLoader
from torchvision import transforms

from ObjectDetNet.retinanet.dataloaders import PredDataset, collater, Resizer, UnNormalizer, Normalizer


def detect(home_path, checkpoint_path):

    class_names_path = os.path.join(home_path, "d.names")
    # compute number of classes
    num_classes = sum(1 for line in open(class_names_path))
    # must have a file to predict on called "predict_on"
    pred_on_path = os.path.join(home_path, 'predict_on')

    #create output path
    checkpoint_name = checkpoint_path.split('.')[0]
    output_path = os.path.join(home_path, 'predictions', checkpoint_name)
    if not os.path.exists(os.path.join(home_path, 'predictions')):
        os.mkdir(os.path.join(home_path, 'predictions'))
    if os.path.exists(output_path):
        raise Exception('there are already predictions for model: ' + checkpoint_name)
    os.mkdir(output_path)

    #copy annotations to predictions
    gt_file = glob.glob(os.path.join(pred_on_path, '*.json'))[0]
    set_name = gt_file.split('/')[-1].split('.')[0].split('_')[1]
    if os.path.exists(gt_file):
        if not os.path.exists(os.path.join(home_path, 'predictions', 'annotations')):
            os.mkdir(os.path.join(home_path, 'predictions', 'annotations'))
        copyfile(gt_file, os.path.join(home_path, 'predictions', 'annotations', gt_file.split('/')[-1]))
    # dataset_val = PredDataset(pred_on_path=pred_on_path, class_list_path=class_names_path,
    #                          transform=transforms.Compose([Normalizer(), Resizer(min_side=608)])) #TODO make resize an input param
    dataset_val = PredDataset(pred_on_path, set_name=set_name,
                        transform=transforms.Compose([Normalizer(), Resizer(min_side=608)]))
    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=None)

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    scales = checkpoint['scales']
    ratios = checkpoint['ratios']

    retinanet = model.resnet152(num_classes=num_classes, scales=scales, ratios=ratios) #TODO: make depth an input parameter
    retinanet.load_state_dict(checkpoint['model'])
    retinanet = retinanet.cuda()
    retinanet.eval()
    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):
        scale = data['scale'][0]
        with torch.no_grad():
            st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.5)[0]
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            detections_list = []
            for j in range(idxs.shape[0]):
                bbox = transformed_anchors[idxs[j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_idx = int(classification[idxs[j]])
                label_name = dataset_val.labels[label_idx]
                score = scores[idxs[j]].item()

                draw_caption(img, (x1, y1, x2, y2), label_name)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)
                # un resize for eval against gt
                bbox /= scale
                bbox.round()
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                detections_list.append([label_name, str(score), str(x1), str(y1), str(x2), str(y2)])
            img_name = dataset_val.coco.dataset['images'][idx]['file_name'].split('.')[0]
            filename = img_name + '.txt'
            filepathname = os.path.join(output_path, filename)
            with open(filepathname, 'w', encoding='utf8') as f:
                for single_det_list in detections_list:
                    for i, x in enumerate(single_det_list):
                        f.write(str(x))
                        f.write(' ')
                    f.write('\n')

            img_save_name = dataset_val.coco.dataset['images'][idx]['file_name']
            save_to_path = os.path.join(output_path, img_save_name)
            cv2.imwrite(save_to_path, img)
            cv2.waitKey(0)
