from sys import path
import pandas as pd
from tqdm import tqdm
import os

def create_sizes_csv(path_to_csv, dir_new_csv, resol) :
    """
    create_sizes_csv permet, à partir du csv au format pascal VOC des bboxs, de creer un nouveau csv qui associe à chaque image un score de largeur et de hauteur.
    Ce score de largeur (resp. hauteur) est obtenu en faisait la moyenne des largeurs de bboxs (resp. hauteur) présentes dans l'image. 
    Puis, on divise le segment [0; W_max] (resp. [0; H_max]) en 2^(la résolution) segments égaux et on affecte par odre croissant un numéro à chaque segment en partant de 0.
    Par exemple, pour une résolution de 1, une largeur moyenne de 124 aura pour score 0, puisque [0; W_max] = [0; 443] donc divisé en 2¹ segments : [0; 221] (auquel appartient 124, d'où son score) et [0; 222].
    Ces grâces à ce csv de scores qu'on va pouvoir extraire un échantillon réprésentatif de la base complète en utilisant l'échantillonnage stratifié.

    :param path_to_csv: str, path du csv au format pascal VOC
    :param resol: int, résolution utilisée tel qu'expliqué plus haut
    :param path_new_csv: str, répertoire où le csv va être créé
    :return: Void
    """

    W_MAX = 443 
    H_MAX = 335

    df = pd.read_csv(path_to_csv)
    df_ship = df[df.xmax == df.xmax] # dataframe des images avec bateau
    sizes_df = pd.DataFrame(columns = ['filename','W_score' , 'H_score'])

    for image_name in tqdm(df_ship['filename'].unique()) :
        image_df = df[df.filename == image_name]
        w = []
        h = []
        for index, row in image_df.iterrows():
            w.append(row['xmax']-row['xmin'])
            h.append(row['ymax']-row['ymin'])
        w_mean = sum(w)/len(w)
        h_mean = sum(h)/len(h)
        w_score = int(2**resol*w_mean/W_MAX)
        h_score = int(2**resol*h_mean/H_MAX)
        new_row = pd.DataFrame.from_dict({'filename': [image_name], 'W_score': [w_score], 'H_score': [h_score]})
        sizes_df = pd.concat([sizes_df, new_row],  ignore_index=True)

    sizes_df.to_csv(os.path.join(dir_new_csv,'boats_sizes_scores_resol'+str(resol)+'.csv'))

    print('csv successfully created at : '+ os.path.join(dir_new_csv,'boats_sizes_scores_resol'+str(resol)+'.csv'))

if __name__ == '__main__' :
    path_to_csv = '/tf/ship_data/train_ship_segmentations_OD.csv'
    dir_new_csv = '/tf/ship_data/boats_sizes'
    resol = 5
    create_sizes_csv(path_to_csv, dir_new_csv, resol)