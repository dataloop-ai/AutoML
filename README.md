# ***ZazuML***

## Getting started

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
i.e. in parallel to each other

**model_priority_space** -  define the model specs that best suits your priorities.

This is a 3 dimensional vector describing your model preferences in a euclidean vector space.

- axis 0 - accuracy
- axis 1 - inference speed
- axis 2 - memory

For example "model_priority_space": [2, 9, 10] indicates a very light but low accuracy model

**task** - i.e. detection vs classification vs instance segmentation

**data** - This is an example of how to run on Coco dataset.

Once you've finished editing your configs.json you're ready to begin!

### Begin model and hyper-parameter search on a machine of your choice
```
python zazutuner.py
```
### Launch model and hyper-parameter search on Kubernetes via our remote Dataloop engine
```
python zazutuner.py --remote 1
```

## Adding your own model to the ***ZazuML*** *model Zoo*
We encourage you to add your own model to the *ZazuML model zoo* and become an 
official contributor to the project. 

### Example of the directory structure of your model
```
├── retinanet
│   ├── __init__.py
│   ├── adapter.py
│   ├── anchors.py
│   ├── dataloaders
│   ├── losses.py
│   ├── model.py
│   ├── oid_dataset.py
│   ├── train_model.py
│   ├── utils.py
```
<br/><br/>    

Every model must have a mandatory ***adapter.py*** file which contains an **AdapterModel** 
class which serves as an adapter between our ***main.py*** and model directory via a range of 
predefined class methods.

### Template for your AdapterModel class
```
class AdapterModel:

    def __init__(self, model_specs, hp_values):
        pass

    def reformat(self):
        pass

    def data_loader(self):
        pass

    def preprocess(self):
        pass

    def build(self):
        pass
        
    def train(self):
        pass
        
    def get_metrics(self):
        pass
```
The "init", "train" and "get_metrics" methods are mandatory methods for running your model. 
The methods are run in the order of the example above, i.e. first the "init" then "reformat" and so on . . 

**reformat** method is where you'd be expected to reformat the input image annotations into a format your
model can handle. Your model is required to handle CSV and Coco styled annotations at the very least.

**get_metrics** method is expected to return a dictionary object in the form of `{'val_accuracy': 0.928}` 
where `0.928` in this example is a python float.

Once you've added your model to the *ZazuML model zoo* you have to append it to the 
*models.json* file so that *ZazuML* knows to call upon it. 

### *Example key value in model.json*

```
  "retinanet": {
    "task": "detection",
    "model_space": {
      "accuracy_rating": 8,
      "speed_rating": 2,
      "memory_rating": 4
    },
    "hp_search_space": [
      {
        "name": "input_size",
        "values": [
          100,
          200,
          300
        ],
        "sampling": null
      },
      {
        "name": "learning_rate",
        "values": [
          1e-4,
          1e-5,
          1e-6
        ],
      }
    ],
    "training_configs": {
      "epochs": 1,
      "depth": 50
    }
  }
```

**hp_search_space** - is for defining hyper-parameters that will over-go optimization 

**training_configs** - is where fixed hyper-parameters are defined

Which parameters will be frozen and which will be optimizable is a design decision 
and will be immutable once the model is pushed to the *ZazuML model zoo*.

**model_space** - is where you define the relative location of your model in a euclidean vector space

**task** - is the defining task of your model, currently you can choose from either 
*classification*, *detection* or *instance segmentation*

The json object key must match the model directory name exactly so that
ZazuML knows what model to call upon, in our example the name of 
both will be ***"retinanet"***.


## Refrences
Some of the code was influenced by [keras-tuner](https://github.com/keras-team/keras-tuner)