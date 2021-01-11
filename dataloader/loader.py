from .kerasDataGenerator import KerasGenerator
from torch.utils.data import DataLoader
from .dataloader import CustomDataset,collater
from .utils import *
class DataGenerator(object):
    def __init__(self, dir_path,annotation_format, framework='pytorch', batch_size=32, dataset='train', target_size=(224, 224),rgb=True, shuffle=True, num_workers=4,annotation_path=None,function_transforms=None, built_in_transforms=None):
        self.dir_path = dir_path
        self.annotation_format=annotation_format
        self.framework = framework
        self.batch_size = batch_size
        self.dataset=dataset
        self.target_size = target_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.annotation_path=annotation_path
        self.rgb=rgb
        self.function_transforms=function_transforms, 
        self.built_in_transforms=built_in_transforms
        if rgb:
            self.n_channels = 3
        else:
            self.n_channels = 1 
        
        if self.framework == 'pytorch':
            self.generator = CustomDataset(dir_path=self.dir_path,annot_format=self.annotation_format,dataset=self.dataset,function_transforms=None, built_in_transforms=None)
            self.generator = DataLoader(dataset=self.generator,batch_size=self.batch_size,shuffle=self.shuffle,num_workers=self.num_workers,collate_fn=collater)
        elif self.framework == 'keras':
            self.generator = KerasGenerator(dir_path=self.dir_path,annotation_format=self.annotation_format,batch_size=self.batch_size,n_channels=self.n_channels,shuffle=self.shuffle,dataset=self.dataset,dim=self.target_size,function_transforms=None, built_in_transforms=None)



    def __iter__(self):
        if self.framework == 'pytorch':
            
            self.generator= iter(self.generator)
            return self.generator
        elif self.framework == 'keras':
            self.generator= iter(self.generator)
            return self.generator

   



