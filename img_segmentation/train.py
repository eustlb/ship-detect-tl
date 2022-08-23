from configparser import Interpolation
from tkinter import image_names
from sacrebleu import metrics
import segmentation_models as sm
from tensorflow.keras.utils import image_dataset_from_directory
sm.set_framework('tf.keras')
import pandas as pd
import os

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class DataBatch(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """
        Renvoie un tuple x, y des images et des masques correspondants pour un batch.
        """
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            y[j] /= 255
        return x, y

def train():
    #
    batch_size = 8
    img_size = (768, 768)
    imgs_dir = '/tf/ship_data/train_v2'
    masks_dir = '/tf/ship_data/masks_only_one_image'
    aug_imgs_dir = '/tf/ship_data/augmented_data/imgs'
    aug_mask_dir = '/tf/ship_data/augmented_data/masks'

    path_train_csv = '/tf/ship_data/annotations/70_80/train_70_80.csv'
    path_aug_csv = '/tf/ship_data/augmented_data/augmented_data_OD.csv'
    path_val_csv = '/tf/ship_data/annotations/70_80/test_70_80.csv'
    df_train = pd.read_csv(path_train_csv)
    df_aug = pd.read_csv(path_aug_csv)
    df_val = pd.read_csv(path_val_csv)

    train_l = df_train['filename'].unique()
    aug_l = df_aug['filename'].unique()
    val_l = df_val['filename'].unique()

    train_input_imgs_paths = [os.path.join(imgs_dir, img_name) for img_name in train_l]
    train_target_imgs_paths = [os.path.join(masks_dir, img_name[:img_name.index('.')]+'_mask'+'.png') for img_name in train_l]

    aug_input_imgs_paths = os.listdir(aug_imgs_dir)
    aug_target_imgs_paths = os.listdir(aug_mask_dir)

    val_input_imgs_paths = [os.path.join(imgs_dir, img_name) for img_name in val_l]
    val_target_imgs_paths = [os.path.join(masks_dir, img_name[:img_name.index('.')]+'_mask'+'.png') for img_name in train_l]


    # load data
    train_gen = DataBatch()
    

    # preprocess input
    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)

    # define model 
    model = sm.FPN(backbone_name='resnet34', input_shape=(None, None, 3), encoder_weights='imagenet', classes=1, activation='sigmoid', pyramid_dropout=.2)
    loss = sm.losses.BinaryCELoss() + 5*sm.losses.JaccardLoss()
    metrics = [sm.IOUScore(),sm.losses.FScore(beta=2)]
    model.compile('Adam', loss=loss,  metrics=metrics)

    # fit model
    # model.fit(
    #     x=x_train,
    #     y=y_train,
    #     epochs=100,
    #     validation_data=(x_val, y_val),
    # )
