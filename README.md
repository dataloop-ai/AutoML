# ZazuML

## Getting started
ZazuML contains both a requirements file and a dockerfile for your convenience.


## Adding your own model to ZazuML model Zoo
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
Every model must have a mandatory ***adapter.py*** file which contains an **AdaptModel 
class** which serves as an adapter between our ***main.py*** via a range of 
predefined class methods.

### Template for your AdaptModel class
```
class AdaptModel:

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

Once you've added your model to the *ZazuML model zoo* you have to append it to the 
*models.json* file so that *ZazuML* knows to call upon it. 

### Example key value in model.json

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