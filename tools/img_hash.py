import pandas as pd
import shutil
from cluster2 import main, rebuild_mosaic

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

df_hash = pd.read_csv('/tf/ship_data/boats_hash.csv')

def find_cluster(boat_h, cluster, know_boats, df_hash):
    know_boats.append(boat_h)
    cluster += [img for img in df_hash[df_hash.BoatHash == boat_h]['ImageId'] if img not in cluster]
    for img_name in cluster:
        for boat_h in [boat_h for boat_h in df_hash[df_hash.ImageId == img_name]['BoatHash']]:
            if boat_h not in know_boats:
                find_cluster(boat_h, cluster, know_boats, df_hash)
    return cluster

cluster = find_cluster(-8053994833403345116, [], [], df_hash)

if __name__ == '__main__':
    main()