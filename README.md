# Ship Detection On Satellite Images

<p align="center">
  <img width="768" height="768" src=./docs/figures/pred_example.png>
</p>

## Table of content

* [General info](#general-info)
* [Folder structure](#folder-structure)
* [Setup](#setup)


## General info

This project deals with the detection of ships on satellite images by deep learning. \
The dataset used is the one provided in 2018 by Airbus during a [Kaggle ship setection competition](https://www.kaggle.com/c/airbus-ship-detection).\
This project was done as part of a 5 months internship at the French Naval Academy Research Institute. \
The dataset is made of 192556 annotated images, with only 42556 containing at least one boat (see ship_detect_tl/tools/explore_data/data_info.ipynb).

## Folder structure 
    .
    ├── data                    # Various files (fonts, etc.) 
    ├── data_augmentation       # Data augmentation scripts 
    ├── data_parsing            # Data parsing scripts, CSV and pickle files
    ├── docs                    # Ducumentation: bibliography, report and tutorials
    ├── img_segmentation        # Image segmentation approach
    ├── masks                   # Scripts to generate binary masks for image segmentation 
    ├── object_detection        # Object detection approach
    ├── tools                   # Tools and utilities
    ├── tutorials               # Tutorials 
    └── README.md

### 1. data_parsing

> Same boats appear on different images (see technical_report.pdf section 1.4). These duplicates are a big issue for the train/val split. This folder contains scripts and CSVs to deal with this problem and insure a train/val split without boats appearing in both datasets. This represents a significant part of the work done.

    .
    ├── ...
    ├── data_parsing            
    │   ├── CSV                 # CSV files explained in description.md (in the CSV folder)
    │   ├── hash                # Scripts implementing the hash approach (see...)
    │   └── mosaics             # Scripts implementing the mosaic approach (see...)
    └── ...

### 2. data_augmentation

Module and notebook to run data augmentation. The [albumentations](https://github.com/albumentations-team/albumentations) library is used.
Here is a distrbution of the sizes (bounding boxes) of the boats contained in this dataset

### 3. object_detection 

#### Setup  

> Here you will find everything you need to train and evaluate the models (bboxes) of the [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).\
The different steps are described in the notebook main.ipynb :


### 4. tools

> This folder contains tools to explore the dataset so as to run predictions with trained models on images.

    .
    ├── ...
    ├── tools            
    │   ├── explore_data        # Scripts & notebooks to get infos on the dataset.
    │   ├── inference           # Scripts to run inference on images using our trained models.
    └── ...

### 5. masks & image_segmentation

...

## Installation

### Data

All the images and CSVs required can be directly downloaded on [Kaggle website](https://www.kaggle.com/competitions/airbus-ship-detection/data).

### Environment

It is necessary to first make sure that Cuda is installed, so as tensorflow and the object detection API ([tensorflow model garden](https://github.com/tensorflow/models)). Nevertheless, it is simplier to create a Docker image with all the dependencies installed and ready to work. See tuto_container_tf_od.pdf in docs folder.

Once inside the container, simply clone this git repository.

It is highly recommended to access the container through VS Code and its remote explorer extension.

Moreover, tools such as tmux to open terminals in the container and nvtop to monitor GPUs can also be really useful.

### Object detection

In the object detection folder (object_detection), follow the steps as described in the notebook main.ipynb. Don't forget to configure the image directory. 
These steps are :
In the object detection folder (object_detection), follow the steps as described in the notebook main.ipynb. Don't forget to configure the image directory. 
These steps are :
- create the train and test tfrecords
- pick a model on [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and download it
- configure the hyperparameters of the model (namely the .config file of the model)
- change the scales and aspect ratios of the anchor generator in the .config file of the models with the values given in the notebook. These values were computed by clustering thanks to the notebook /tools/explore_data/EDA_of_bbox.ipynb
- configure CUDA_VISIBLE_DEVICES env variable (in container terminals) if you want to run multiple GPU training.
- generate commands for training and validation
- paste theses command in container terminals (where CUDA_VISIBLE_DEVICES env variable was first set).
- run tensorboard to check results

## A few results

Here are a few interesting results. The idea was to verify empirically some of the conclusions found when reading papers. 
All the models were trained during 25k steps over six GPUs, with a batch size of 48 (synchronous distributed training, tensorflow MirroredStrategy). 
They all used momentum optimizer with a coefficient of momentum of 0.9 and cosine decay learning rate schedule, with base value of 0.04, 500 warmup steps for a total of 25k steps.
When transfer learning was used, the pretrained weights were those obtained after training on COCO, initialized from Imagenet classification checkpoint and during trainging no layer were freezed. 

### Interests of transfer learning 

As mentionned above, models were first trained on COCO dataset (and even before ImageNet dataset). Nevertheless, one might think that these datasets as so different from Kaggle's one that transfer learning may not be really useful. Indeed, COCO dataset is composed of 91 categories, of which only the boat one is interesting here. Moreover, objects in this dataset come with many angles and are quite big on the images while in our dataset, the angle is always vertical (satellite view) and boats quite small (a majority of around 20 pixels wide which represent only around 3% of the image width). 

The main improvement to be expected is in detecting "small object". A "small object", according to the precision metric, is an object with a max rectangle area of 32x32 pixels. 
[Stastitics on COCO dataset](https://arxiv.org/pdf/1902.07296v1.pdf) (see table 2 in the article) show that the majority (more thant 40%) of the objects are small. For the train dataset we used (see train\_test\_split/70\_80), more than 50% of the boats have a max rectangle area below 27x27 pixels. For this reason, we can expect that knowledge learned, particularly by the feature extractor (here ResNet50), will have the stronger impact for detecting small object.

<p align="center">
  <img width="600" height="400" src=./docs/figures/mAP.png>
</p>

The above graph empirically demonstrates a well-known result : transfer learning speeds up training and improve the performance of your deep learning model. Lets know 


<p float="left">
  <img src=./docs/figures/mAP_large.png width="400" />
  <img src=./docs/figures/mAP_medium.png width="400" /> 
  <img src=./docs/figures/mAP_small.png width="400" />
</p>


We can see small object detection training is the one that has been accelerated the most, with an accuracy after 2k steps : 5 times higher for small boats against 1.9 times higher for large boats and 2 times higher for medium boats.


### Data augmentation vs. deeper network

>"Clearly we see that changing augmentation can be as, if not more,
powerful than changing around the underlying architectural components." ([Learning Data Augmentation Strategies for Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720562.pdf))