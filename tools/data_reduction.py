import os
import shutil
import pandas as pd

paths = os.listdir('/tf/ship_data/train_v2/train')

def get_labelled_images(path_to_csv):
    """
    Return the list of labelled images names
    """
    df = pd.read_csv(path_to_csv)
    return df[df.EncodedPixels == df.EncodedPixels]['ImageId'].unique()

i, j = 0, 0
image_names = get_labelled_images('/tf/ship_data/train_ship_segmentations_v2.csv')
paths = os.listdir('/tf/ship_data/train_v2/train')
while i<1000:
    while paths[j] not in image_names:
        j+=1
    src = '/tf/ship_data/train_v2/train/' + paths[j]
    dest = '/tf/ship_data_lite/'+ os.path.basename(paths[j])
    shutil.copyfile(src, dest)
    i+=1
    j+=1