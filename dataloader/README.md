
  

  

# Dataloader

### Object Outline

```
class customDataset(Dataset):
    def __init__(self, img_path, format, functions_transforms=None, built_in_augmentations=None):

    def __getitem__(self, index):

    def __len__(self):

    def visualize(self, save_path):

    @property
    def num_classes(self):

```

- *format* is string of yolo or coco.
- *_functions_of_transform_* is a list with single/variety of transform functions.
- *_built_in_transform_* is a list of string with single/variety of augmentations in the library.

##
* __**\_\_init\_\_**__: Initaialize dataset path, transforms.

* __**\_\_getitem\_\_**__: preprocess the image and annotations and save as a dictionary. i.e. {"img":img,"ann":ann,"scale": scale}

* __**\_\_len\_\_**__: return the length of the dataset.

* __**visualize**__: draw the image with bounding box and save it to the path

* __**num_classes**__: the number of catergories to the dataset

  

  

### Project Usage Example

#### [In]:

```python

from data import available_augmentations, CustomDataset

dataset = CustomDataset("your/data/folder/path", "yolo", function_transforms = [Rotate(),Resize()])
print(len(dataset))
print(dataset[5])
print(available_augmentations[0,4])
dataloader_iterator = iter(dataset)
data = next(dataloader_iterator)
print(data)
data = next(dataloader_iterator)
print(data)

```

#### [Out]:

```python
5123
{"img":img, "ann":ann, "scale": scale}  #dataset[5]
array([[ 1, 2, 3, 4],
       [ 5, 6, 7, 8],
       [ 9, 10, 11, 12],
       [ 13, 14, 15, 16]])
{"img":img, "ann":ann, "scale": scale}  #dataset[0]
{"img":img, "ann":ann, "scale": scale}  #dataset[1]
```

# Built-in Augmentations Usage Example

Augmentations we provide below , you can simply use ```augmentation_list()```to check out the list.

Before you use the library, don't forget to import it. ```from dataloader import custom_transforms```
``` 
augmentations= ['Translate_Y',
                'Translate_Y_BBoxes',
                'Translate_X',
                'Translate_X_BBoxes',
                'CutOut',
                'CutOut_BBoxes',
                'Rotate',
                'ShearX',
                'ShearX_BBoxes',
                'ShearY',
                'ShearY_BBoxes',
                'Equalize',
                'Equalize_BBoxes',
                'Solarize',
                'Solarize_BBoxes',
                'Color',
                'Color_BBoxes',
                'FlipLR'
                ]
```
``` python 
    augmentations = [('CutOut', 0.5, 0.9), ('Rotate', 0.5, 0.9),('Color', 1, 0.9)]
    y = CustomDataset('/your/dataset/path', 'yolo', built_in_transforms=augmentations)
    y[3].visualize('/save/path/')
```




On this example, one of the augmentations will be random chosen to apply on the dataset. Below is the result.``` ('CutOut', 0.5, 0.9)``` For the parameters, in the tuple, the first one is the name of transform, the second one is the probability which is within [0,1], and the last field is the strength which is within [0,1].
###
![1](picture1.png)
 
