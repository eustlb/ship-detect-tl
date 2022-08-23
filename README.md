# Ship Detection On Satellite Images

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
    ├── img_segmentation        # Image segmentation approach
    ├── masks                   # Scripts to generate binary masks for image segmentation 
    ├── object_detection        # Object detection approach
    ├── tools                   # Tools and utilities
    ├── tutorials               # Tutorials 
    └── README.md

### 1. data_parsing

> Same boats appear on different images (see...). These duplicates are a big issue for the train/val split. This folder contains scripts and CSVs to deal with this problem and insure a train/val split without boats appearing in both datasets.

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

> It is necessary to first make sure that Cuda is installed, so as tensorflow and the object detection API ([tensorflow model garden](https://github.com/tensorflow/models)). Nevertheless, it is simplier to create a Docker image with all the dependencies installed and ready to work. See tutorial in the tutorial folder. 

Here you will find everything you need to train and evaluate the models (bboxes) of the [tensorflow model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).\
The different steps are described in the notebook main.ipynb :


### 4. tools

> Here you will find tools to explore the dataset so as to run predictions with trained models on images.

    .
    ├── ...
    ├── tools            
    │   ├── explore_data        # Scripts & notebooks to get infos on the dataset.
    │   ├── inference           # Scripts to run inference on images using our trained models.
    └── ...

### 5. masks & image_segmentation

...

## A few results
