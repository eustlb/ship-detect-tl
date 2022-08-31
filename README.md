# Ship Detection On Satellite Images

<p align="center">
  <img width="768" height="768" src=./docs/figures/pred_example.png>
</p>

## Table of content

* [General info](#general-info)
* [Folder structure](#folder-structure)
* [Installation](#installation)
* [Results](#results)
* [Generalisation](#generalisation)


## General info

This project deals with the detection of ships on satellite images by deep learning. \
The dataset used is the one provided in 2018 by Airbus during a [Kaggle Ship Detection Competition](https://www.kaggle.com/c/airbus-ship-detection).\
This project was done as part of a 5-month internship at the French Naval Academy Research Institute. \
The dataset is made of 192556 annotated images, with only 42556 containing at least one boat (see tools/explore_data/data_info.ipynb).

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

> Same boats appear on different images (see docs/technical_report.pdf section 1.4). These duplicates are a big issue for the train/val split. This folder contains scripts and CSVs to deal with this problem and ensure a train/val split without boats appearing in both datasets. This represents a significant part of the work done and is described in the report docs/technical_report.pdf. 

    .
    ├── ...
    ├── data_parsing            
    │   ├── CSV                 # CSV files explained in description.md (in the CSV folder)
    │   ├── hash                # Hash approach (docs/technical_report.pdf sect. 2.1-2.2)
    │   └── mosaics             # Mosaic approach (docs/technical_report.pdf sect. 2.4)
    └── ...

### 2. data_augmentation

Module and notebook to run data augmentation. The [Albumentations](https://github.com/albumentations-team/albumentations) library is used.

### 3. object_detection 

#### Setup  

In this folder, you will find everything you need to train and evaluate the models (bboxes) of the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).\
The different steps are described in the notebook object_detection/main.ipynp.


### 4. tools

> This folder contains tools to explore the dataset and to run predictions with trained models on images.

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

It is necessary to first make sure that Cuda is installed, so as TensorFlow and the object detection API ([TensorFlow Model Garden](https://github.com/tensorflow/models)). Nevertheless, it is simpler to create a Docker image with all the dependencies installed and ready to work. See docs/tuto_container_tf_od.pdf.

Once inside the container, simply clone this git repository.

It is highly recommended to access the container through VS Code and its remote explorer extension.

Moreover, tools such as Tmux to open terminals in the container and nvtop to monitor GPUs can also be useful.

### Object detection

In the object detection folder (object_detection), follow the steps as described in the notebook main.ipynb. Don't forget to configure the image directory. 
These steps are :
In the object detection folder (object_detection), follow the steps as described in the notebook main.ipynb. Don't forget to configure the image directory. 
These steps are :
- create the train and test TFRecords
- pick a model on [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) and download it
- configure the hyperparameters of the model (namely the .config file of the model)
- change the scales and aspect ratios of the anchor generator in the .config file of the models with the values given in the notebook. These values were computed by clustering thanks to the notebook tools/explore_data/EDA_of_bbox.ipynb
- configure the CUDA_VISIBLE_DEVICES env variable (in container terminals) if you want to run multiple GPU training
- generate commands for training and validation
- paste these commands in container terminals (where CUDA_VISIBLE_DEVICES env variable was first set)
- run TensorBoard to check the results

## Results

Here are a few interesting results. The idea was to verify empirically some of the conclusions found when reading papers. 
All the models were trained during 25k steps over six GPUs, with a batch size of 48 (synchronous distributed training, TensorFlow MirroredStrategy). 
They all used a momentum optimizer with a coefficient of momentum of 0.9 and a cosine decay learning rate schedule, with a base value of 0.04, 500 warmup steps for a total of 25k steps.
When transfer learning was used, the pre-trained weights were those obtained after training on COCO, initialized from ImageNet classification checkpoints and during training no layers were frozen. 

### Benefits of transfer learning 

As mentioned above, models were first trained on the COCO dataset (and even before the ImageNet dataset). Nevertheless, one might think that these datasets are so different from Kaggle's one that transfer learning may not be really useful. Indeed, the COCO dataset is composed of 91 categories, of which only one (boats) is interesting here. Objects in this dataset come with many angles while in our dataset, the angle is always vertical (satellite view) and boats quite small (a majority of around 20 pixels wide which represent only around 3% of the image width). But, between starting from randomly initialized weights and starting for pre-trained ones, one would at first glance have nothing to lose by choosing the transfer learning option.

The main improvement to be expected is in detecting "small objects". A "small object", according to the precision metric, is an object with a max rectangle area of 32x32 pixels. 
[Stastitics on the COCO dataset](https://arxiv.org/pdf/1902.07296v1.pdf) (see table 2 in the article) show that the majority (more than 40%) of the objects are small. For the training dataset, we used (see train\_test\_split/70\_80), more than 50% of the boats have a max rectangle area below 27x27 pixels. For this reason, we can expect that this knowledge learned, particularly by the feature extractor (here ResNet50), will have a greater impact on detecting small objects.

<p align="center">
  <img width="600" height="400" src=./docs/figures/mAP.png>
</p>

The above graph empirically demonstrates a well-known result: transfer learning speeds up training and improves the performance of your deep learning model.

<p align="center">
  <img width="450" height="300" src=./docs/figures/mAP_large.png>
</p>
<p align="center">
  <img width="450" height="300" src=./docs/figures/mAP_medium.png>
</p>
<p align="center">
  <img width="450" height="300" src=./docs/figures/mAP_small.png>
</p>

We can see that small object detection training is the one that has been accelerated the most, with an accuracy after 2k steps: 5 times higher for small boats against 1.9 times higher for large boats and 2 times higher for medium boats.


### Data augmentation vs. Deeper network

>"Clearly we see that changing augmentation can be as, if not more,
powerful than changing around the underlying architectural components." ([Learning Data Augmentation Strategies for Object Detection](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123720562.pdf))

Here we are going to change de ResNet50 backbone for a deeper one: ResNet101.

<p align="center">
  <img width="600" height="400" src=./docs/figures/aug_mAP.png>
</p>

We can see that, in both case, our data augmentation strategy led to an improvement of about 2% mAP while changing the backbone for a deeper one led to a 5% increase in mAP. Verifying the statement quoted above is difficult as we decided to train both models for the same numbers of steps while the deeper one may have needed some more steps for its precision metric to reach its top. Likewise, we used exactly the same hyperparameters knowing that other parameters could have led to better results. Moreover, such results really depend on the data augmentation strategy. In our case, the chosen strategy is inspired by the one used in the paper. Nevertheless, we need to keep in mind that in this paper they use AutoAugment which may have led to a more optimal strategy. 

## Generalisation

The generalizability of our trained model (Faster R-CNN, ResNet101 backbone, data augmentation - around 56% mAP) has been tested on drone marina footage. Knowing that the training base is pretty different from such images, where the seagulls look more like the boats of our training data, our model seems to have a good generalization capacity

https://user-images.githubusercontent.com/94853470/187454349-deeef5d8-f7b0-4909-b02e-26c96eb5aa8b.mp4
