## Getting started

First thing to do is . . .  

### *pull & run the Docker Image*
```
docker run --rm -it --init  --runtime=nvidia  --ipc=host  -e NVIDIA_VISIBLE_DEVICES=0 buffalonoam/zazu-image:0.3 bash
```
Be sure to update the nvidia-devices flag!

### *clone the repo*
```
git clone https://github.com/dataloop-ai/ZazuML.git

```

### *download tiny coco dataset*
```
mkdir data
cd data
git clone https://github.com/dataloop-ai/tiny_coco.git
cd ../ZazuML
```

### *model & hyper-parameter search*
```
python zazu.py --search
```

### *predict*
```
python zazu.py --predict
```
