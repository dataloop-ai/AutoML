

# Dataloader
### Object Outline
```
    class customDataset(Dataset):
        def __init__(self, img_path, format, tramsform="None", functions_transforms, built_in_transforms):

        def __getitem__(self, index):
          
        def __len__(self):
        
        def visualize(self, save_path):

        @property
        def num_classes(self):
        
```

*format is string of yolo or coco.*
 *functions_of_transform* is a list with single/variety of transform functions. 
 *built_in_args* is a list of string with single/variety of argumentations in the library.

* __init__: Initaialize dataset path, transforms.
* __getitem__: preprocess the image and annotations and save as a dictionary. i.e. {"img":img,"ann":ann,"scale": scale}
* __len__: return the length of the dataset.
* __visualize__: draw the image with bounding box and save it to the path
* __num_classes__: the number of catergories to the dataset




### Project Usage Example
#### [In]:
```
from data import available_argumentations, CustomDataset

    dataset = CustomDataset("your/data/folder/path", "yolo", [Rotate(),Resize()], available_argumentations[0,4])
    dataloader_iterator = iter(dataset)
    data = next(dataloader_iterator)
	
```



