import numpy as np 
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

def rleToMaskPNG(rleString,height,width,mask_dir,mask_name):

  rows,cols = height,width
  img = np.zeros(rows*cols,dtype=np.uint8)
  rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
  rlePairs = np.array(rleNumbers).reshape(-1,2)
  for index,length in rlePairs:
    index -= 1
    img[index:index+length] = 255

  img = img.reshape(cols,rows)
  img = img.T
  im = Image.fromarray(img)
  im.save(os.path.join(mask_dir, mask_name))

def imgToMaskPNG(rleStringL,height,width,mask_dir,mask_name):

  rows,cols = height,width
  img = np.zeros(rows*cols,dtype=np.uint8)
  for rleString in rleStringL:
    rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
    rlePairs = np.array(rleNumbers).reshape(-1,2)
    for index,length in rlePairs:
      index -= 1
      img[index:index+length] = 255

  img = img.reshape(cols,rows)
  img = img.T
  im = Image.fromarray(img)
  im.save(os.path.join(mask_dir, mask_name))

def main1():

  # empty image when no boats
  empty_mask = np.zeros(768*768,dtype=np.uint8)
  empty_mask = empty_mask.reshape(768,768)
  empty_mask = empty_mask.T
  empty_mask = Image.fromarray(empty_mask)

  mask_dir = '/tf/ship_data/masks'
  df_origin =  pd.read_csv('/tf/ship_data/train_ship_segmentations_v2.csv')

  for img_name in tqdm(df_origin['ImageId'], total=df_origin.shape[0]):
    rle_l = [rle for rle in df_origin[df_origin.ImageId == img_name]['EncodedPixels']]
    i = 1 
    for rle in rle_l:
      mask_name = img_name[:img_name.index('.')]+'_mask_'+str(i)+'.png'
      i+=1
      if rle == rle:
        rleToMaskPNG(rle, 768, 768, mask_dir, mask_name)

def main2():

  mask_dir = '/tf/ship_data/masks_only_one_image'
  df_origin =  pd.read_csv('/tf/ship_data/train_ship_segmentations_v2.csv')

  # for img_name in tqdm(df_origin['ImageId'][], total=df_origin.shape[0]):
  #   rle_l = [rle for rle in df_origin[df_origin.ImageId == img_name]['EncodedPixels']]
  #   mask_name = img_name[:img_name.index('.')]+'_mask'+'.png'

  for img_name in ['000194a2d.jpg']:
    rle_l = [rle for rle in df_origin[df_origin.ImageId == img_name]['EncodedPixels']]
    mask_name = img_name[:img_name.index('.')]+'_mask'+'.png'
  
  imgToMaskPNG(rle_l, 768, 768, mask_dir, mask_name)


if __name__ == '__main__':

  main2()

  