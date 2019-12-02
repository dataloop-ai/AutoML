# ***ZazuML***

## Getting started

First thing to do is pull and run the docker image
```
docker run -it buffalonoam/zazu-image:0.1 bash
```
On the other hand you can `pip install -r requirements.txt` file and hope for the best.

clone the repository and its submodules
```
git clone --recurse-submodules https://noamrosenberg@bitbucket.org/dataloop-ai/zazuml.git
```
<br/><br/>   
The next thing to do is edit the configs.json file

### *configs.json example*
```
{
  "max_trials": 5,
  "max_instances_at_once": 2,
  "model_priority_space": [10, 0, 0],
  "task": "detection",
  "data": {
    "coco_path": "/home/noam/data/coco",
    "annotation_type": "coco"
  }
}
```
**max_trials** - defines the maximum total number of trials that will be tested

**max_instances_at_once** - defines the number of trials that will run simultaneously, 
i.e. in parallel to each other and must be smaller than the number of available gpus.

**model_priority_space** -  define the model specs that best suits your priorities.

This is a 3 dimensional vector describing your model preferences in a euclidean vector space.

- axis 0 - accuracy
- axis 1 - inference speed
- axis 2 - memory

For example "model_priority_space": [2, 9, 10] indicates a very light but low accuracy model

**task** - i.e. detection vs classification vs instance segmentation

**data** - This is an example of how to run on Coco dataset.

Once you've finished editing your configs.json you're ready to begin!

### Begin model & hyper-parameter search on a machine of your choice
```
python zazutuner.py
```
### Launch search on Kubernetes via our remote Dataloop engine
```
python zazutuner.py --remote 1
```

## Refrences
Some of the code was influenced by [keras-tuner](https://github.com/keras-team/keras-tuner)