## Getting started

First thing to do is . . .  

### *pull & run the Docker Image*
```
docker run --rm -it --gpus all  buffalonoam/zazu-image:0.9 bash
```
About the docker image: Should work over all Cuda 10.* systems, this API is for docker 19.03 +

### *If you don't use docker, clone the repo and data*
```
git clone https://github.com/dataloop-ai/ZazuML.git
mkdir data
cd data
git clone https://github.com/NoamRosenberg/tiny_coco.git
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
