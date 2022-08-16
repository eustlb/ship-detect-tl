from sys import path
import albumentations as A
import pandas as pd
import cv2
import os
import numpy as np

def rle2bbox(rle, shape):
    '''
    taken from : https://www.kaggle.com/code/eigrad/convert-rle-to-bounding-box-x0-y0-x1-y1/notebook
    '''
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))  # an array of (start, length) pairs
    a[:,0] -= 1  # `start` is 1-indexed
    
    y0 = a[:,0] % shape[0]
    y1 = y0 + a[:,1]
    if np.any(y1 > shape[0]):
        # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)
    
    x0 = a[:,0] // shape[0]
    x1 = (a[:,0] + a[:,1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)
    
    if x1 > shape[1]:
        # just went out of the image dimensions
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
            x1, shape[1]
        ))

    return x0, y0, x1, y1

class image:

    def __init__(self, img_dir, img_name, mask_dir, path_to_csv):
        self.img = cv2.imread(os.path.join(img_dir,img_name))
        self.name = img_name
        df = pd.read_csv(path_to_csv)
        self.rleStringL = [rle for rle in df[df.ImageId == img_name]['EncodedPixels']]
        self.mask_name = img_name[:img_name.index('.')]+'_mask'+'.png'
        self.mask = cv2.imread(os.path.join(mask_dir,self.mask_name))
        self.height, self.width, self.channels = self.img.shape
        self.bboxes = [rle2bbox(rle,(self.height, self.width)) for rle in self.rleStringL]

    def save(self, saving_dir, mask_dir=None):
        cv2.imwrite(os.path.join(saving_dir, self.name), self.img)
        if mask_dir is not None:
            cv2.imwrite(os.path.join(mask_dir, self.mask_name), self.mask)


def augment(img_dir, img_name, mask_dir, saving_dir, path_to_csv):

    # 4 rotations, 4 flips :
    angles = [90,180,270]
    for i in range(3):
        img_obj = image(img_dir,img_name,mask_dir,path_to_csv)

        img_obj.img = np.rot90(img_obj.img, k=i+1)
        img_obj.mask = np.rot90(img_obj.mask, k=i+1)



        img_obj.name = img_obj.name[:img_obj.name.rfind('.')]+'_rot'+str(angles[i])+'.png'
        img_obj.mask_name = img_obj.mask_name[:img_obj.mask_name.rfind('.')]+'_rot'+str(angles[i])+'.png'

        img_obj.save(saving_dir, mask_dir)



if __name__ == '__main__':

    img_dir = '/tf/ship_data/train_v2'
    img_name = '000194a2d.jpg'
    mask_dir = '/tf/ship_data/masks_only_one_image'
    saving_dir = ('/tf')
    path_to_csv = '/tf/ship_data/train_ship_segmentations_v2.csv'

    img_obj = image(img_dir,img_name,mask_dir,path_to_csv)
    print(img_obj.bboxes)

    # augment(img_dir, img_name, mask_dir, saving_dir, path_to_csv)


        





