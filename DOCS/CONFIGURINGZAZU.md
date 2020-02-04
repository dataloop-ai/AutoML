The next thing to do is edit the configs.json file

### *configs.json example*
```
{
  "max_trials": 1,
  "max_instances_at_once": 1,
  "model_priority_space": [10, 0, 0],
  "task": "detection",
  "data": {
    "home_path": "/home/noam/data/coco",
    "annotation_type": "coco",
    "dataset_name": "2017"
  }
}
```
**max_trials** - defines the maximum total number of trials that will be tested

**max_instances_at_once** - defines the number of trials that will run simultaneously, 
i.e. in parallel to each other and must be smaller than the number of available gpus.

**model_priority_space** -  define the model specs that best suits your priorities.

This is a 3 dimensional vector describing your model preferences in a euclidean vector space.
Each element can occupy the space [0,10). 

- axis 0 - accuracy
- axis 1 - inference speed
- axis 2 - memory

For example "model_priority_space": [2, 9, 10] indicates a very light but low accuracy model

**task** - i.e. detection vs classification vs instance segmentation (we currently only support detection)

**data** - This is an example of how to run on a Coco styled dataset.


### ***Once you've finished editing your configs.json you're ready to begin!***