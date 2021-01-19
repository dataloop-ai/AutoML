
from torch.utils.data import DataLoader
from .dataloader import CustomDataset,collater
from .utils import *
import random
from keras.utils import to_categorical

def dataGenerator(dir_path,annotation_format, framework='pytorch', batch_size=32, dataset='train',shuffle=True, num_workers=4,annotation_path=None,function_transforms=None, built_in_transforms=None):
    
    generator = CustomDataset(dir_path=dir_path,annot_format=annotation_format,dataset=dataset,annotation_path=annotation_path,function_transforms=function_transforms, built_in_transforms=built_in_transforms)
   
    loader = DataLoader(dataset=generator,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,collate_fn=collater)
       
    iterator = iter(loader)
    num_data=len(generator)
    num_batchs=num_data/batch_size

    count = 0
    while count <num_batchs:
                
        if framework=='pytorch':
            yield next(iterator)
        elif framework == 'keras':
            data = next(iterator)
            image = data.image.permute(0, 3, 2, 1)  
            image = np.array(image)
           
            if annotation_format == 'csv' or annotation_format == 'txt':
                label=to_categorical(data.label,num_classes=generator.num_classes)
                label =np.array(label)
              
                yield (image, label)
            else:
                
                annotation = np.array(data.annotation)
               
                yield (image, annotation)

        count +=1










