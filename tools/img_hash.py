from asyncio import as_completed
import pandas as pd
import shutil
from cluster2 import main, rebuild_mosaic, expand_cluster
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

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

    :param path_to_csv: str, path du csv de départ, csv d'origine fourni dans la base kaggle.
    :param path_new_csv: str, path du nouveau csv qui va être créé et contenant les hash des bateaux.
    :return: Void.
    """
    hash_dict = {'ImageId':[], 'BoatHash':[]}

    df_rle = pd.read_csv(path_to_csv)
    df_rle_boats = df_rle[df_rle.EncodedPixels == df_rle.EncodedPixels] # dataframme des images contenant au moins un bateau

    for img_name in tqdm(df_rle_boats['ImageId'].unique()):
        for rle in df_rle_boats[df_rle_boats.ImageId == img_name]['EncodedPixels']:
            hash_dict['ImageId'].append(img_name)
            hash_dict['BoatHash'].append(hash(normalized_rle(rle)))

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

if __name__ == '__main__':

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