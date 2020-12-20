
  

  

# Dataloader

### Object Outline

```
class customDataset(Dataset):
    def __init__(self, img_path, format, function_transforms=None, built_in_transform=None):

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

```

from data import available_augmentations, CustomDataset

dataset = CustomDataset("your/data/folder/path", "yolo", [Rotate(),Resize()], available_augmentations[0,4])
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

```
5123
{"img":img, "ann":ann, "scale": scale}  #dataset[5]
array([[ 1, 2, 3, 4],
       [ 5, 6, 7, 8],
       [ 9, 10, 11, 12],
       [ 13, 14, 15, 16]])
{"img":img, "ann":ann, "scale": scale}  #dataset[0]
{"img":img, "ann":ann, "scale": scale}  #dataset[1]
```
