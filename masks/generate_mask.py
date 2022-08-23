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

def generate_mask_rle(img_l, mask_dir, path_to_csv):
  """
  Créer les images masque à partir des images présentes dans img_dir.
  Les masques enrégistrés sont au format png.
  Il y aura un masque par bateau, c'est à dire un masque pour chaque rle de l'image !

  :param img_l: list, liste des images dont on souhaite créer les masques.
  :param mask_dir: str, répertoire où seront créés les masques
  :param path_to_csv: str, chemin du csv original qui comporte les rle.
  """

  # empty image when no boats
  empty_mask = np.zeros(768*768,dtype=np.uint8)
  empty_mask = empty_mask.reshape(768,768)
  empty_mask = empty_mask.T
  empty_mask = Image.fromarray(empty_mask)

  df_origin =  pd.read_csv(path_to_csv)

  for img_name in tqdm(img_l):
    rle_l = [rle for rle in df_origin[df_origin.ImageId == img_name]['EncodedPixels']]
    i = 1 
    for rle in rle_l:
      mask_name = img_name[:img_name.index('.')]+'_mask_'+str(i)+'.png'
      i+=1
      if rle == rle:
        rleToMaskPNG(rle, 768, 768, mask_dir, mask_name)

def generate_mask_img(img_l, mask_dir, path_to_csv):
  """
  Créer les images masque à partir des images présentes dans img_dir.
  Les masques enrégistrés sont au format png.
  Il y aura un masque par image, l'ensemble des bateaux contenus dans l'image apparaîtra sur le même masque.

  :param img_l: list, liste des images dont on souhaite créer les masques.
  :param mask_dir: str, répertoire où seront créés les masques
  :param path_to_csv: str, chemin du csv original qui comporte les rle.
  """

  df_origin =  pd.read_csv(path_to_csv)

  for img_name in tqdm(img_l):
    rle_l = [rle for rle in df_origin[df_origin.ImageId == img_name]['EncodedPixels']]
    mask_name = img_name[:img_name.index('.')]+'_mask'+'.png'
    imgToMaskPNG(rle_l, 768, 768, mask_dir, mask_name)


if __name__ == '__main__':
  print()

  