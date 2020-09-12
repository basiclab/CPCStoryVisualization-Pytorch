# CPCStoryVisualization-Pytorch
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/character-preserving-coherent-story/story-visualization-on-pororo)](https://paperswithcode.com/sota/story-visualization-on-pororo?p=character-preserving-coherent-story)

![](https://raw.githubusercontent.com/basiclab/CPCStoryVisualization-Pytorch/master/images/introduction4.jpg)

```
Objects in pictures should so be arranged as by their very position to tell their own story.
           - Johann Wolfgang von Goethe (1749-1832)
```

Author: [@yunzhusong](http://github.com/yunzhusong), [@theblackcat102](http://github.com/theblackcat102), [@redman0226](http://github.com/redman0226), Huiao-Han Lu, Hong-Han Shuai

[Paper(PDF)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620018.pdf)

Code implementation for Character-Preserving Coherent Story Visualization


## Datasets

CLEVR with segmentation mask, 13755 sequence of images, generate using [Clevr-for-StoryGAN](https://github.com/theblackcat102/Clevr-for-StoryGAN)

```
images/
    CLEVR_new_013754_1.png
    CLEVR_new_013754_1_mask.png
    CLEVR_new_013754_2.png
    CLEVR_new_013754_2_mask.png
    CLEVR_new_013754_3.png
    CLEVR_new_013754_3_mask.png
    CLEVR_new_013754_4.png
    CLEVR_new_013754_4_mask.png
```

Download [link](https://drive.google.com/drive/folders/1zRT5TCpHTzY32v0YTi9n9-L4c0md0CAK?usp=sharing)

Pororo, original pororo datasets with self labeled segmentation mask of the character.

## Pretrained model

weights for pretrained model can be download at [link](https://drive.google.com/drive/folders/1Oy-Npt19hYvrGAB_u5c_XYnuBsoBu34b?usp=sharing)

Model parameters is the default used in this repository

## Setup

### Setup environment

```
    virtualenv -p python3 env
    source env/bin/activate
    pip install -r requirements.txt
```


Create a environment file .env with the following line:

```
DATAPATH = "PATH TO pororoSV/"
```

### File placement 

pororoSV should contain SceneDialogues/  ( where gif files reside ) and *.npy files

1. setup image for video dataloader

```
    python preprocess_pororo.py
```

2. Segment results should placed under DATAPATH and rename as img_segment 



3. Change related directory PORORO_PATH, etc


4. Try to run main_pororo.py and good luck

### Tensorboard

```
    tensorboard --logdir output/ --host 0.0.0.0 --port 6009
```




