import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import numpy as np

H = 768 
W = 768

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

def normalized_rle(rle):
    """
    Prend en argument un rle au format str et renvoie le rle normalisé, 
    c'est à dire conservant la forme définie par le rle mais placé en haut à gauche de l'image.
    Ainsi peut importe sa position dans l'image, un bateau aura toujours le même rle normalisé.

    :param rle: str, rle d'un bateau sur une image.
    :return new_rle: str, rle normalisé correspondant
    """
    pixels = [int(el) for el in rle.split(' ')[::2]]
    lenghts = [int(el) for el in rle.split(' ')[1::2]]
    leftmost_pix = pixels[0] # le pixel le plus à gauche est le premier
    upper_pix_step = min([pix%768 for pix in pixels])

    new_pixels = []
    # on décale tous les pixels vers le haut et vers la gauche
    for pix in pixels:
        new_value = pix-768*(leftmost_pix//768) # vers le haut
        new_value = new_value-upper_pix_step
        new_pixels.append(new_value)

    new_rle = ' '.join([str(item) for pair in zip(new_pixels, lenghts) for item in pair])
    return new_rle

def hash_boats_rle(path_to_csv, path_new_csv):
    """
    Prends le CSV de départ, c'est-a-dire qui associe à une image le(s) rle du bateau(x) qu'elle contient, 
    et crée un nouveau CSV, analogue au premier mais ou chaque rle à été transformé d'abord en son rle normalisé,
    puis en un entier grâce à une fonction de hashage (afin d'être ensuite comparables). 
    Seront également présent dans le CSV le rle (afin de savoir de quel bateau et sur quelle image on parle),
    la largeur (W) et hauteur (H) du bateau.

    :param path_to_csv: str, path du csv de départ, csv d'origine fourni dans la base kaggle.
    :param path_new_csv: str, path du nouveau csv qui va être créé et contenant les hash des bateaux.
    :return: Void.
    """
    hash_dict = {'ImageId':[], 'BoatRLE':[], 'BoatHash':[], 'W':[], 'H':[]}

    df_rle = pd.read_csv(path_to_csv)
    df_rle_boats = df_rle[df_rle.EncodedPixels == df_rle.EncodedPixels] # dataframme des images contenant au moins un bateau

    for img_name in tqdm(df_rle_boats['ImageId'].unique()):
        for rle in df_rle_boats[df_rle_boats.ImageId == img_name]['EncodedPixels']:
            hash_dict['ImageId'].append(img_name)
            hash_dict['BoatRLE'].append(rle)
            hash_dict['BoatHash'].append(hash(normalized_rle(rle)))
            xmin, ymin, xmax, ymax = rle2bbox(rle, (H, W))
            hash_dict['W'].append(xmax-xmin)
            hash_dict['H'].append(ymax-ymin)

    df = pd.DataFrame.from_dict(hash_dict)
    pd.DataFrame.to_csv(df, path_new_csv)

def imgs_per_b_csv(path_hash_csv, path_new_csv):
    """
    Genère le CSV de toutes les images qui contiennent un certain bateau (identifié par son hash).

    :param path_hash_csv: str, chemin du csv généré par hash_boats_rle
    :param path_new_csv: str, chemin du nouveau csv qui va être créé.
    :return: Void.
    """
    df_hash = pd.read_csv(path_hash_csv)
    boats = df_hash['BoatHash'].unique()
    boats_h_dict = {'BoatHash':[], 'ImageIds':[]}
    for boat in tqdm(boats) : 
        imgs = [img for img in df_hash[df_hash.BoatHash == boat]['ImageId']]
        imgs_str = ' '.join(imgs)
        boats_h_dict['BoatHash'].append(boat)
        boats_h_dict['ImageIds'].append(imgs_str)
    df = pd.DataFrame.from_dict(boats_h_dict)
    pd.DataFrame.to_csv(df, path_new_csv)

def find_clusters(path_csv):
    """
    Renvoie les clusters formés en traitant le problème comme un problème réseau.
    On considère les images contenant un même bateau comme des sommets reliés d'un graphe.
    Ainsi, il s'agit de former des clusters contenant des sommets connectés entre eux et isolés du reste des sommets.
    Cela revient à chercher à fusionner des listes qui partagent des éléments.

    :param path_csv: str, chemin du csv formé par imgs_per_b_csv
    :return out: list, liste des clusters (qui sont des listes de noms d'images) formés.
    """
    # creer la liste des imgs par bateau : 
    df = pd.read_csv(path_csv)
    l = []
    for index,row in df.iterrows():
        l.append(row['ImageIds'].split(' '))

    out = []

    # algorithme qui permet de fusionner des listes qui partagent des éléments. 
    # pris sûr : https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements
    while len(l)>0:
        first, *rest = l
        first = set(first)

        lf = -1
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2

        out.append(first)
        l = rest
        if len(l)%1000 < 10:
            print(len(l))

    return out

def main1():
    """
    Permet de calculer par multiprocessing (ici sur 96 coeurs) l'ensemble des clusters que l'on peut former à partir de la base de départ.
    Principe : 
    Sera éxectué sur chaque coeur :
    - On appelle la fonction find_cluster sur un boat_h. On obient ainsi un cluster d'images.
    - Toutes les images de ce cluster sont supprimés du dataframe de départ permettant de chercher les clusters.
    - On recommence avec le boat_suivant, jusqu'à avoir tout traité. 
    """

    df_hash = pd.read_csv('/tf/ship_data/boats_hash.csv')
    num_workers = 96
    clusters = []
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = [executor.submit(find_cluster, boat_h, [], [], df_hash) for boat_h in df_hash['BoatHash'].unique()[:num_workers]]

    i = 0
    prev_len, j = len(df_hash['BoatHash'].unique()), 0 # utilisé pour faire des sauvegardes du travail effectué tous les 10000 clusters créés

    while len(df_hash['BoatHash'].unique()) != 0:
        for future in futures:
            if future.done():
    
                cluster = future.result()
                clusters.append(cluster)

                # supprimer du dataframe les bateaux associées aux images du cluster si cela n'a pas déjà été fait par un process concurrent afin d'éviter d'y chercher pour rien 
                for img_name in cluster:
                    df_hash.drop(df_hash[df_hash.ImageId == img_name].index, inplace=True)
                    
                # on relance un nouveau processus
                index = futures.index(future)
                futures.remove(future)
                boat_h = df_hash['BoatHash'].unique()[0]
                futures.insert(index,executor.submit(find_cluster, boat_h, [], [], df_hash))

                i += 1
                print(len(df_hash['BoatHash'].unique()))

                if i%100 == 0 :
                    print(len(df_hash['BoatHash'].unique()))
        
        if prev_len-len(df_hash['BoatHash'].unique())>1000:
            f = open('/tf/clusters_h/clusters_h_'+str(j)+'.pkl', "wb") 
            pickle.dump(clusters, f)
            f.close()
            j+=1
            prev_len = len(df_hash['BoatHash'].unique())

def main2():
    path_csv = '/tf/ship_data/imgs_per_boats.csv'
    clusters = find_clusters(path_csv)
    for cluster in clusters:
        l = []
        for el in cluster:
            l.append(el)
        cluster = l

    print(clusters[0][0])
    # sauvegarde de la liste des clusters sur le disque
    f = open('/tf/clusters_h/clusters_h.pkl', "wb") 
    pickle.dump(clusters, f)
    f.close()

if __name__ == '__main__':
    path_to_csv = '/tf/ship_data/train_ship_segmentations_v2.csv'
    path_new_csv = '/tf/ship_data/find_duplicates/hash/boats_hash.csv'
    # hash_boats_rle(path_to_csv, path_new_csv)

    rle = '86727 2 87493 4 88261 4 89030 3 89798 4 90566 4 91334 4 92103 3 92871 1'

    print(hash(normalized_rle(rle)))
