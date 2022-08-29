# Ship Detection On Satellite Images

<p align="center">
  <img width="768" height="768" src=./docs/pred_example.png>
</p>

## Table of content

* [General info](#general-info)
* [Folder structure](#folder-structure)
* [Setup](#setup)


## General info

This project deals with the detection of ships on satellite images by deep learning. \
The dataset used is the one provided in 2018 by Airbus during a [Kaggle ship setection competition](https://www.kaggle.com/c/airbus-ship-detection).\
This project was done as part of a 5 months internship at the French Naval Academy Research Institute.

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

> Same boats appear on different images (see technical_report.pdf section 1.4). These duplicates are a big issue for the train/val split. This folder contains scripts and CSVs to deal with this problem and insure a train/val split without boats appearing in both datasets.

    .
    ├── ...
    ├── data_parsing            
    │   ├── CSV                 # CSV files explained in description.md (in the CSV folder)
    │   ├── hash                # Scripts implementing the hash approach (see...)
    │   └── mosaics             # Scripts implementing the mosaic approach (see...)
    └── ...

### 2. data_augmentation

Module and notebook to run data augmentation. The [albumentations](https://github.com/albumentations-team/albumentations) library is used.

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
