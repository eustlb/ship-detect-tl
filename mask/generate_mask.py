import numpy as np 
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

def rleToMaskPNG(img_name,rle_l,height,width,mask_dir):
  rows,cols = height,width
  img = np.zeros(rows*cols,dtype=np.uint8)
  
  if rle_l[0] == rle_l[0]: # On vérifie que l'image contient bien un bateau, c'est-à-dire qu'il n'y pas de rle NaN 
    rlePairs_l = []
    for rleString in rle_l :
      rleNumbers = [int(numstring) for numstring in rleString.split(' ')]
      rlePairs = np.array(rleNumbers).reshape(-1,2)
      rlePairs_l.append(rlePairs)
    for rlePairs in rlePairs_l:
      for index,length in rlePairs:
        index -= 1
        img[index:index+length] = 255

  img = img.reshape(cols,rows)
  img = img.T
  im = Image.fromarray(img)
  mask_name = img_name[:img_name.index('.')]+'_mask'+'.png'
  im.save(os.path.join(mask_dir, mask_name))

if __name__ == '__main__':

  # empty image when no boats
  empty_mask = np.zeros(768*768,dtype=np.uint8)
  empty_mask = empty_mask.reshape(768,768)
  empty_mask = empty_mask.T
  empty_mask = Image.fromarray(empty_mask)

  mask_dir = '/tf/ship_data/masks'
  df_origin =  pd.read_csv('/tf/ship_data/train_ship_segmentations_v2.csv')
  for img_name in tqdm(df_origin['ImageId'], total=df_origin.shape[0]):
    rle_l = [rle for rle in df_origin[df_origin.ImageId == img_name]['EncodedPixels']]
    rleToMaskPNG(img_name, rle_l, 768, 768, mask_dir)

