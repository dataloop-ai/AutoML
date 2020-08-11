## Adding your own model to the ***ObjectDetNet***
We encourage you to add your own model to the *ZazuML model zoo* and become a contributor to the project. 

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
class and a ***predict*** function, which serves as an adapter between ***ZazuML*** and our ***ZaZoo*** 

### Template for your AdapterModel class
```
class AdapterModel:

    def load(self, checkpoint_path='checkpoint.pt'):
        raise NotImplementedError

    def reformat(self):
        pass

    def preprocess(self, batch):
        return batch

    def build(self):
        pass

    def train(self):
        pass

    def get_checkpoint(self):
        pass

    @property
    def checkpoint_path(self):
        raise NotImplementedError

    def save(self, save_path='checkpoint.pt'):
        raise NotImplementedError

    def predict(self, checkpoint_path='checkpoint.pt', output_dir='checkpoint0'):
        raise NotImplementedError

    def predict_single_image(self, image_path, checkpoint_path='checkpoint.pt'):
        raise NotImplementedError
```

**load** method is where you pass all the important information to your model 

- device - gpu index to be specified to all parameters and operations requiring gpu in this specific trial
- model_specs - contains model configurations and information relevant to the location of your data and annotation type
- hp_values - are the final hyper parameter values passed to this specific trial

**reformat** method is where you'd be expected to reformat the input image annotations into a format your
model can handle. Your model is required to handle CSV and Coco styled annotations at the very least.

**get_checkpoint** method is expected to return a pytorch styled .pt file

**get_metrics** method is expected to return a dictionary object in the form of `{'val_accuracy': 0.928}` 
where `0.928` in this example is a python float

**predict** function receives the path to your data storage directory as well as to a checkpoint.pt file
