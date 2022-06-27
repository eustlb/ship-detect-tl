import os
import cv2
from importlib_metadata import csv
import pandas as pd
from tqdm import tqdm

def hash_img(im_path):
    h = []
    step = 256
    img = cv2.imread(im_path)
    for x_grid in range(0,768,256):
        for y_grid in range(0,768,256):
            crop = img[x_grid: x_grid+step, y_grid: y_grid+step]
            h.append(hash(crop.tobytes()))
    return h

def create_hash_csv(img_dir, csv_path):
    names = os.listdir(img_dir)
    print('Hashing images crops...')
    rows_list = [[img_name]+hash_img(os.path.join(img_dir, img_name)) for img_name in tqdm(names)]
    df = pd.DataFrame(rows_list,columns=['filename',0,1,2,3,4,5,6,7,8])
    df.to_csv(csv_path, index=False)
    print('CSV created at: '+ csv_path)

def find_duplicates(img_name, compare_imgs, hash_csv_path):
    """
    :param compare_imgs: list, Liste des noms d'images dans laquelle on cherche les duplicatas
    """
    hash_df = pd.read_csv(hash_csv_path)
    h1 = [hash_df.loc[hash_df['filename'] == img_name][str(i)].values[0] for i in range(9)]
    duplicates = []

    for name in tqdm(compare_imgs):
        h2 = [hash_df.loc[hash_df['filename'] == name][str(i)].values[0] for i in range(9)]
        if set(h1) & set(h2):
            duplicates.append(name)
            compare_imgs.remove(name)

    return duplicates
        

def recompose(path_csv):
    df = pd.read_csv(path_csv)
    duplicates_list = [string.split(',') for string in df.iloc[:,0].tolist()]
    for i in range(len(duplicates_list)):
        duplicates = duplicates_list[i]

def main():
    from multiprocessing import Pool, cpu_count
    print(f'starting computations on {cpu_count()} cores')






if __name__ == '__main__':
    img_dir = '/tf/ship_data/train_v2'
    csv_path = '/tf/ship_data/boats_info/hash.csv'
    create_hash_csv(img_dir, csv_path)

    import multiprocessing





