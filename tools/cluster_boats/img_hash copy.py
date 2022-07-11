from asyncio import as_completed
import pandas as pd
import shutil
from cluster2 import main, rebuild_mosaic, expand_cluster
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import networkx 
from networkx.algorithms.components.connected import connected_components

def normalized_rle(rle):
    """
    Prend en argument un rle au format str et renvoie le rle normalisé, c'est à dire la même forme définie mais placée en haut à gauche de l'image
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
    hash_dict = {'ImageId':[], 'BoatHash':[]}

    df_rle = pd.read_csv(path_to_csv)
    df_rle_boats = df_rle[df_rle.EncodedPixels == df_rle.EncodedPixels] # dataframme des images contenant au moins un bateau

    for img_name in tqdm(df_rle_boats['ImageId'].unique()):
        for rle in df_rle_boats[df_rle_boats.ImageId == img_name]['EncodedPixels']:
            hash_dict['ImageId'].append(img_name)
            hash_dict['BoatHash'].append(hash(normalized_rle(rle)))

    df = pd.DataFrame.from_dict(hash_dict)
    pd.DataFrame.to_csv(df, path_new_csv)

def find_cluster(boat_h, cluster, know_boats, df_hash):
    """
    Utilisé de façon récursive. Les listes de départ cluster et know_boats doivent être vides au premier appel de la fonction.
    Principe :
    Lorsque appelée sur un certain boat_h, elle ajoute au cluster toutes les images qui contiennent ce bateau et qui ne sont pas encore dans le cluster.
    Puis pour chacune des images du cluster, on regarde chaque bateau contenu par ces images et on reappel notre fonction pour chacun de nos bateau avec les listes ainsi formées.
    
    """
    know_boats.append(boat_h)
    cluster += [img for img in df_hash[df_hash.BoatHash == boat_h]['ImageId'] if img not in cluster]
    for img_name in cluster:
        for boat_h in [boat_h for boat_h in df_hash[df_hash.ImageId == img_name]['BoatHash']]:
            if boat_h not in know_boats:
                find_cluster(boat_h, cluster, know_boats, df_hash)
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

if __name__ == '__main__':
    path_csv = '/tf/ship_data/imgs_per_boats.csv'
    clusters = find_clusters(path_csv)
    for cluster in clusters:
        l = []
        for el in cluster:
            l.append(l)
        cluster = l

    print(clusters[0][0])
    

    # sauvegarde de la liste des clusters sur le disque
    # f = open('/tf/clusters_h/clusters_h.pkl', "wb") 
    # pickle.dump(clusters, f)
    # f.close()
