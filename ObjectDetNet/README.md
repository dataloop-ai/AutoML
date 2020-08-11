## Getting started with the ***ObjectDetNet***
*ObjectDetNet* is an easy, flexible, open-source object detection framework which allows you to easily train, resume & 
prototype training sessions, run inference and flexibly work with checkpoints in a production grade environment.

## Quick Start
Copy and paste this into your command line
```
#run in docker 
docker run --rm -it --init  --runtime=nvidia  --ipc=host  -e NVIDIA_VISIBLE_DEVICES=0 buffalonoam/zazu-image:0.3 bash

mkdir data
cd data
git clone https://github.com/dataloop-ai/tiny_coco.git
cd ..
git clone https://github.com/dataloop-ai/ObjectDetNet.git
cd ObjectDetNet
python main.py --train
```
After training just run:
```
python main.py --predict 
# OR 
python main.py --predict_single
# to predict a single item
```

To change the data you run on or the parameters of your model just update the example_checkpoint.pt file!

At the core of the *ObjectDetNet* framework is the ***checkpoint object***. The ***checkpoint object*** is a json, 
pt or json styled file to be loaded into python as a dictionary. ***Checkpoint objects*** aren't just used for training, 
but also necessary for running inference. Bellow is an example of how a checkpoint object might look.
```
├── {} devices
│   ├── {} gpu_index
│       ├── 0
├── {} model_specs
│   ├── {} name
│       ├── retinanet
│   ├── {} training_configs
│       ├── {} depth
│           ├── 152
│       ├── {} input_size
│       ├── {} learning_rate
│   ├── {} data
│       ├── {} home_path
│       ├── {} annotation_type
│           ├── coco
│       ├── {} dataset_name
├── {} hp_values
│       ├── {} learning_rate
│       ├── {} tuner/epochs
│       ├── {} tuner/initial_epoch
├── {} labels
│       ├── {} 0
│           ├── Rodent
│       ├── {} 1
│       ├── {} 2
├── {} metrics
│       ├── {} val_accuracy
│           ├── 0.834
├── {} model
├── {} optimizer
├── {} scheduler
├── {} epoch
│       ├── 18
```
For training your checkpoint dictionary must have the following keys:
- device - gpu index for which to convert all tensors
- model_specs - contains 3 fields 
    1. name
    2. training_configs
    3. data

To resume training you'll also need:
- model - contains state of model weights
- optimizer - contains state of optimizer
- scheduler - contains state of scheduler
- epoch - to know what epoch to start from

To run inference your checkpoint will need:
- model_specs
- labels


If you'd like to customize by adding your own model, check out [Adding a Model](./ADDMODEL.md)

### Feel free to reach out with any questions
WeChat: BuffaloNoam   
Line: buffalonoam   
WhatsApp: +972524226459  

## Refrences
Thank you to these repositories for their contributions to the ***ObjectDetNet***

- Yhenon's [pytorch-retinanet](https://github.com/yhenon/pytorch-retinanet)
- Qqwweee's [keras-yolo3](https://github.com/qqwweee/keras-yolo3)