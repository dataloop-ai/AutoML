![Logo](./images/ZazuML.jpeg)
<br/><br/>   

This is an easy open-source AutoML framework for object detection. This project contains a 
architeture search + hyper-parameter search + augmentations search, as well as trial manager. 
An ML Pipeline taylor made for simple integration into your project or code.

<br></br>
<br></br>

![](./images/zazu_eng.png)

### *Algorithms*

| *Augmentations Search* | *Hyper-Parameter Search* | *Architecture Search*
| :----:         |     :----:      |    :----:      |
|  <ul><li>Fast-Autoaugment</li></ul>    | <ul><li>Random Search</li><li>HyperBand</li></ul>     | <ul><li>EfficientDet</li><li>EndToEndTransformers</li></ul>  |
### *Super EASY to use!*

![](./images/running_zazu_search2.gif)  

## *Why ZazuML?*
You might be building an ML pipeline to avoid model performance degeneration, or maybe you're just too lazy to download, 
debug, and tune your own model. Either way, you shouldn't be focusing your efforts on simple things like detection. There's
a whole world out there for you to explore, give your hand at trajectory prediction or action recognition and let *ZazuML*
free you up from the boring stuff.

## *Launch Remotely*
ZazuML's *REMOTE* feature is designed so that you can run search, train and predict anywhere. No GPU's or complex Cloud APIs.
It's easy to implement into your python script or run in your terminal.

### *via Terminal* 
![](./images/zazu_remote_search.gif)

### *via iPython*
![](./images/zazu_via_sdk.gif)

- Get started with [Quick Start](DOCS/GETTINGSTARTED.md)
- Read up on [Configuring ZazuML](DOCS/CONFIGURINGZAZU.md)
- Launch [ZazuML remotely](DOCS/REMOTEZAZU.md)
- Take a look [Under The Hood](DOCS/UNDERTHEHOOD.md)
- Customize by [adding your own models](DOCS/ADDINGMODELS.md)


## *TO DO*

- Increase search space
- NAS to replace some of the HP search
- Intelligent Losses to replace some of the HP search

## *Contact*

If you're interested in becoming a collaborator or just have some questions, feel free to contact me at:

WeChat: BuffaloNoam   
Line: buffalonoam   
WhatsApp: +972524226459   

## *Refrences*

Some of the code was inspired by [keras-tuner](https://github.com/keras-team/keras-tuner)
