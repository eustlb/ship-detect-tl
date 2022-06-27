import os
import cv2
from tqdm import tqdm

dir = '/tf/imgs_crops'

def build_crops(img_path):
    crops = []
    step = 256
    img = cv2.imread(img_path)
    for x_grid in range(0,768,256):
        for y_grid in range(0,768,256):
            crop = img[x_grid: x_grid+step, y_grid: y_grid+step]
            crop = cv2.resize(crop,(6,6),interpolation = cv2.INTER_AREA)
            crops.append(crop)
    return crops

for img_name in tqdm(os.listdir('/tf/ship_data/train_v2')):
    crops = build_crops('/tf/ship_data/train_v2/'+img_name)
    i = 1
    for crop in crops :
        cv2.imwrite('/tf/imgs_crops/'+img_name[:img_name.index('.')]+'_'+str(i)+'_'+'.png',crop)
        i+=1

