# ZazuML

## Running ZazuML



## Adding your own model to ZazuML model Zoo
We encourage you to add your own model to ZazuML model zoo and become an 
official contributor to the project. The structure of your model directory should 
be as follows:
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
Every model must have a mandatory adapter.py file 
which contains an AdaptModel class which serves as an adapter between our main.py 
via a range of predefined class methods as exemplified below:
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

Once you've added your model to the ZazuML model zoo you have to append it to the 
models.json file ZazuML so that Zazu knows to call upon it. 

At this point you should define the "task", "model_space", "hp_search_space", 
and "training_configs" fields using the following template:

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
The "hp_search_space" field is for defining hyper-parameters that will over-go 
optimization, while "training_configs" is where set hyper-parameters 
are defined. The json object key must match the model directory name exactly so that
ZazuML knows what model to call upon, in our example the name of the model directory will 
be "retinanet".


### Refrences
Some of the code was influenced by [keras-tuner](https://github.com/keras-team/keras-tuner)