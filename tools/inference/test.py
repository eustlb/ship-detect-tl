import shutil
import pickle
import os
from matplotlib.transforms import Bbox
from tqdm import tqdm
import pandas as pd

import numpy as np

import os
import cv2 
import tensorflow as tf
import numpy as np
import pandas as pd
from object_detection.builders import model_builder
from object_detection.utils import config_util
from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageDraw
from tqdm import tqdm

def rebuild_mosaic(cluster):

    i_max = max([el[1][0] for el in cluster])
    i_min = min([el[1][0] for el in cluster])
    j_max = max([el[1][1] for el in cluster])
    j_min = min([el[1][1] for el in cluster])

    grey_box = np.full((256, 256, 3), (56,62,66), dtype=np.uint8)

    img_tile = [[grey_box for i in range(i_max-i_min+3)] for j in range (j_max-j_min+3)]

    for el in cluster :
        i,j = el[1][0]-i_min, -(el[1][1]-j_max)
        img = cv2.imread('/tf/ship_data/train_v2/' + el[0])
        for x in range(3):
            for y in range(3):
                crop = img[x*256: (x+1)*256, y*256: (y+1)*256]
                img_tile[j+x][i+y] = crop

    mosaic = cv2.vconcat([cv2.hconcat(list_h) for list_h in img_tile]) 

    for el in cluster:
        if el[1][0]==0 and el[1][1]==0:
            base_name = el[0][:el[0].index('.')]
    
    cv2.imwrite('/tf/mosaic/'+str(base_name)+'_mosaic.png',mosaic)

def save_image_bbox(img_path, saving_path):
    """
    Dessine sur une image des bboxs avec leurs scores.

    :param img_path: str, path de l'image à éditer
    :param saving_path: str, path où sera enregistrée l'image 
    :param bboxs: list, liste de bboxs au format [[ymin, xmin, ymax, xmax]]
    :param scores: list, liste des scores au format [0.8]
    :return: Void.
    """
    image_truth = Image.open(img_path)
    h, w = 768, 768
    draw_truth = ImageDraw.Draw(image_truth)

    labels_df = pd.read_csv('/tf/ship_data/train_ship_segmentations_OD.csv')

    detections_bboxs = []
    detections_scores = []

    image_name = os.path.basename(img_path)
    df = labels_df.loc[labels_df['filename'] == image_name]

    for index, row in df.iterrows():
        l = [row['ymin']/h, row['xmin']/w, row['ymax']/h, row['xmax']/w]
        detections_bboxs.append(l)
        detections_scores.append(1.)

    for i in range(len(detections_bboxs)) :
        x0 = detections_bboxs[i][1]*w
        y0 = detections_bboxs[i][0]*h
        x1 = detections_bboxs[i][3]*w
        y1 = detections_bboxs[i][2]*h
        font_file = "/tf/ship_detect_tl/data/BebasNeue-Regular.ttf"
        font_size = 12
        font = ImageFont.truetype(font_file, font_size)
        text = str(round(detections_scores[i]*100))+'%'
        height = font.getsize(text)[1]
        draw_truth.rectangle([x0,y0,x1,y1], outline='#20d200')
        draw_truth.text((x0,y0-height-3), text, font = font, fill=(32, 210, 0))
    image_truth.save(saving_path, "PNG")


l1 = ['d0109dfaf.jpg', '00578738f.jpg', '2d6caffd4.jpg', '70dcb6c38.jpg']
l2 = ['001bcf222.jpg', 'b60cd7528.jpg','208d78bf6.jpg']
l = ['001bcf222.jpg', '70dcb6c38.jpg']
# l = l1 + l2

# for img in l :
#     save_image_bbox('/tf/ship_data/train_v2/'+img, '/tf/'+img)




# 
































# df_rle = pd.read_csv('/tf/ship_data/train_ship_segmentations_v2.csv')
# df_rle_boats = df_rle[df_rle.EncodedPixels == df_rle.EncodedPixels] # dataframme des images contenant au moins un bateau

# for img_name in l:
#         for rle in df_rle_boats[df_rle_boats.ImageId == img_name]['EncodedPixels']:
#             print(img_name, hash(normalized_rle(rle)))
#             print(rle)
#             bbox = rle2bbox(rle, (768, 768))
#             x0 = bbox[1]
#             y0 = bbox[0]
#             x1 = bbox[3]
#             y1 = bbox[2]
#             print(x1-x0, y1-y0)
#             bbox = [rle2bbox(rle, (768, 768))]
#             print('\n')


with open("/tf/ship_data/mosaics/clusters/clusters.pkl", "rb") as fp:   # Unpickling
        clusters = pickle.load(fp)

i = 0 
l = []
for cluster in clusters:
    if len(cluster) == 1:
        i+=1
        l.append(cluster[0][0])



data_label = pd.read_csv('/tf/ship_data/train_ship_segmentations_v2.csv')
boats = data_label[data_label.EncodedPixels == data_label.EncodedPixels]['ImageId'].unique()

a = 0
for img1 in tqdm(l) :
    if img1 not in boats: 
        a += 1
print(a)


# img_list = []
# cluster1 = []
# for cluster in clusters:
#     for img in cluster:
#         if len(img_list)<len(cluster) and img[0] == '2d6caffd4.jpg':
#             img_list = [img[0] for img in cluster]
#             cluster1 = cluster
    
# rebuild_mosaic(cluster1)


# print(img_list)

# i = 0
# for img_name in img_list :
#     shutil.copy('/tf/ship_data/train_v2/'+img_name, '/tf/tests/'+img_name)
#     i+=1



