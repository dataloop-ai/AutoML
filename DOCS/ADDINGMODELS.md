##Adding Models to models

In order to make *ZazuML* aware of a (model)[https://github.com/dataloop-ai/zoo] you have to append it to the 
*models.json* file.

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
        "name": "learning_rate",
        "values": [
          5e-4,
          1e-5,
          5e-5
        ]
      },
      {
        "name": "anchor_scales",
        "values": [
          [1, 1.189207115002721, 1.4142135623730951],
          [1, 1.2599210498948732, 1.5874010519681994],
          [1, 1.5, 2.0]
        ]
      }
    ],
    "training_configs": {
      "epochs": 100,
      "depth": 50,
      "input_size": 608,
      "learning_rate": 1e-5,
      "anchor_scales": [1, 1.2599210498948732, 1.5874010519681994]
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
