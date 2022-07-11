from sys import path
from cv2 import IMWRITE_TIFF_COMPRESSION, compare
import pandas as pd 
from tqdm import tqdm
import os

hash_df = pd.read_csv('/tf/ship_data/boats_info/hash.csv')

# hash_df = hash_df.set_index('Unnamed: 0')
# hash_df.index.names = ['filename']


# csv_path = '/tf/ship_data/boats_info/hash.csv'

# hash_df.to_csv(csv_path, index=False)

# print(hash_df.columns)

def find_duplicates(img_name, compare_imgs, hash_csv_path):
    """
    :param compare_imgs: list, Liste des noms d'images dans laquelle on cherche les duplicatas
    """
    hash_df = pd.read_csv(hash_csv_path)
    h1 = [hash_df.loc[hash_df['filename'] == img_name][str(i)].values[0] for i in range(9)]
    print(h1)
    duplicates = []

    for name in tqdm(compare_imgs):
        h2 = [hash_df.loc[hash_df['filename'] == name][str(i)].values[0] for i in range(9)]
        print(h2)
        if set(h1) & set(h2):
            print(h2)
            duplicates.append(name)
            compare_imgs.remove(name)

    return duplicates

# duplicates = find_duplicates('31f0f5cd2.jpg',['09e29c7f7.jpg'],'/tf/ship_data/boats_info/hash.csv')

# print(duplicates)

import cv2

im1 = cv2.imread('/tf/dupli_samples/2/1.jpg')
im2 = cv2.imread('/tf/dupli_samples/2/2.jpg')
im3 = cv2.imread('/tf/dupli_samples/2/3.jpg')
im4 = cv2.imread('/tf/dupli_samples/2/4.jpg')
im5 = cv2.imread('/tf/dupli_samples/2/5.jpg')
im6 = cv2.imread('/tf/dupli_samples/2/6.jpg')
im7 = cv2.imread('/tf/dupli_samples/2/7.jpg')
im8 = cv2.imread('/tf/dupli_samples/2/8.jpg')
im9 = cv2.imread('/tf/dupli_samples/2/9.jpg')

im_list = [im1, im2, im3, im4, im5, im6, im7, im8, im9]

crop1 = im1[256:512,256:512]
crop2 = im2[256:512, 512:768]

def build_crops(im1,im2):
    """
    Prends ...
    """
    crops = []
    step = 256
    for x_grid in range(0,768,256):
        crop1 = im1[x_grid: x_grid+step, 0:256]
        crop2 = im2[x_grid: x_grid+step, 257:513]
        crops.append([crop1, crop2])
    return crops

def diff(crop1,crop2):
    n = 256*256*3
    m = 0
    for i in range(crop2.shape[0]):
        for j in range(crop2.shape[1]):
            # if crop1[i,j,0] != crop2[i,j,0] or crop1[i,j,1] != crop2[i,j,1] or crop1[i,j,2] != crop2[i,j,2]:
            #     # print(i+512,j)
            #     m+=1
            print(crop1[i,j])
            if crop1[i,j] != crop2[i,j]:
                # print(i+512,j)
                m+=1
    return(m/n)

def save_diff(crop1,crop2, path):
    for i in range(crop2.shape[0]):
        for j in range(crop2.shape[1]):
            if crop1[i,j,0] != crop2[i,j,0] or crop1[i,j,1] != crop2[i,j,1] or crop1[i,j,2] != crop2[i,j,2]:
                crop1[i,j,0] = 0
                crop1[i,j,1] = 0
                crop1[i,j,2] = 0
    cv2.imwrite(path,crop1)

def build_diffs(crops, dir):
    for i in range(len(crops)):
        save_diff(crops[i][0], crops[i][1], os.path.join(dir,'crop'+str(i)+'.jpg'))
        print(diff(crops[i][0], crops[i][1]))


def overlay(im1_path, im2_path):
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)

def compare_crops(crop1,crop2):
    if diff(crop1,crop2) > 0.05:
        return False
    else:
        return True

crops = build_crops(im1, im2)

def find_duplicates(img_name, crop_number, img_list):
    duplicates = []
    crop1 = cv2.imread('/tf/imgs_crops/'+img_name[:img_name.index('.')]+'_'+str(crop_number)+'_'+'.png')
    for img_name_2 in tqdm(img_list):
        for i in range(9):
            crop2 = cv2.imread('/tf/imgs_crops/'+img_name_2[:img_name_2.index('.')]+'_'+str(i+1)+'_'+'.png')
            dist = cv2.norm(crop1 - crop2, cv2.NORM_L2)
            if dist < 50 :
                duplicates.append(img_name_2)
    return duplicates


import time

if __name__ == '__main__':


    img_name = '4e3393ed5.jpg'
    img_list = os.listdir('/tf/ship_data/train_v2')

    start = time.time()
    duplicates = find_duplicates(img_name,5,img_list)
    end = time.time()

    print(duplicates)

    
    print(end-start)
    

