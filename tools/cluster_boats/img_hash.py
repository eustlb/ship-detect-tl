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
    Seront également présent dans le CSV le rle (afin de savoir de quel bateau on parle),
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

def find_cluster(boat_h, cluster, known_boats, df_hash):
    """
    Utilisé de façon récursive. Les listes de départ cluster et know_boats doivent être vides au premier appel de la fonction.
    Principe :
    Lorsque appelée sur un certain boat_h, elle ajoute au cluster toutes les images qui contiennent ce bateau et qui ne sont pas encore dans le cluster.
    Puis pour chacune des images du cluster, on regarde chaque bateau contenu par ces images et on appelle de nouveau notre fonction pour chacun de nos bateau avec les listes ainsi formées.
    
    :param boat_h: int, hash du rle normalisé du bateau. Premier bateau à partir duquel va être construit le cluster.
    :param cluster: list, liste images présentes dans un même cluster. Est d'abord vide au premier appel de la fonction.
    :param known_boats: list, liste des bateaux (hash) connus comme étant dans ce cluster.
    :param df_hash: dataframe tiré du CSV construit par la fonction hash_boats_rle.
    :return cluster: list, liste des images (noms) présentes dans un même cluster construit à partir du premier bateau.
    """
    known_boats.append(boat_h)
    cluster += [img for img in df_hash[df_hash.BoatHash == boat_h]['ImageId'] if img not in cluster]
    for img_name in cluster:
        for boat_h in [boat_h for boat_h in df_hash[df_hash.ImageId == img_name]['BoatHash']]:
            if boat_h not in known_boats:
                find_cluster(boat_h, cluster, known_boats, df_hash)
    return cluster

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current  

def find_clusters(path_csv):
    # creer la liste des imgs par bateau : 
    df = pd.read_csv(path_csv)
    l = []
    for index,row in df.iterrows():
        l.append(row['ImageIds'].split(' '))

    out = []

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
    df_hash = pd.read_csv('/tf/ship_data/boats_hash.csv')
    # df_hash = df_hash[df_hash.ImageId in df_hash['ImageId'].unique()[:10]]
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
            l.append(l)
        cluster = l

    print(clusters[0][0])
    # sauvegarde de la liste des clusters sur le disque
    f = open('/tf/clusters_h/clusters_h.pkl', "wb") 
    pickle.dump(clusters, f)
    f.close()

if __name__ == '__main__':
    path_to_csv = '/tf/ship_data/train_ship_segmentations_v2.csv'
    path_new_csv = '/tf/ship_data/find_duplicates/hash/boats_hashV2.csv'
    hash_boats_rle(path_to_csv, path_new_csv)