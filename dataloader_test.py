from dataloader import *


def test1():
    # yolo with function_transforms
    y = CustomDataset('/root/data/mask_dataset', 'yolo')
    print("before: ", y[3])
    print("image : ", y[3].image.shape)
    print("image type:", type(y[3].image))

    print("image : ", y[3].annot.shape)
    print("image type:", type(y[3].annot))
    y[3].visualize()

    y = CustomDataset('/root/data/mask_dataset', 'yolo', function_transforms=[CutOut(10), ])
    print("after: ", y[3])

    print("image shape: ", y[3].image.shape)
    print("annotation type:", type(y[3].image))

    print("image shape: ", y[3].annot.shape)
    print("annotation type:", type(y[3].annot))
    y[3].visualize()


def test2():
    y = CustomDataset('/root/data/car_dataset/train', 'coco')
    print("before: ", y[3])
    print("image shape: ", y[3].image.shape)
    print("annotation type:", type(y[3].image))

    print("image : ", y[3].annot.shape)
    print("image type:", type(y[3].annot))

    y = CustomDataset('/root/data/car_dataset/train', 'coco', function_transforms=[CutOut(10), ])
    print("after: ", y[3])

    print("image shape: ", y[3].image.shape)
    print("annotation type:", type(y[3].image))

    print("image shape: ", y[3].annot.shape)
    print("annotation type:", type(y[3].annot))


def test3():
    # visualize

    augmentations = [ ('Rotate', 1, 0.9),('Rotate', 1, 0.9),('Rotate', 1, 0.9)]
    c = CustomDataset('/root/data/car_dataset/train', 'coco', built_in_transforms=augmentations)
    c[8].visualize(instance=False,alpha=0.5)



def test4():
    # buil in transform  before and after
    y = CustomDataset('/root/data//mask_dataset', 'yolo')
    y[3].visualize()
    print(y[4])
    augmentations = [('CutOut', 0.5, 0.9), ('Rotate', 0.5, 0.9),('Color', 1, 0.9)]
    y = CustomDataset('/root/data/mask_dataset', 'yolo', built_in_transforms=augmentations)
    y[3].visualize('/root/')
    print(y[4])

def test5():

    c = CustomDataset(image_path='/root/data/car_dataset/train', data_format='coco')

    c[1].visualize(instance=False,alpha=0.5)
    c[2].visualize(instance=False,alpha=0.9)
    c[3].visualize(instance=True,alpha=0.9)
    c[4].visualize(instance=True,alpha=0.9)
    c[5].visualize(instance=True,alpha=0.9)
    c[6].visualize(instance=True,alpha=0.9)
    c[8].visualize(instance=True, alpha=0.9)


def test6():
    t = TrainFile(file_path='/Users/yi-chu/Downloads/dog',file_format='csv')
    print(t)
    t = TrainFile(file_path='/Users/yi-chu/Downloads/dog',file_format='txt')
    print(t)






if __name__ == '__main__':
  test5()
