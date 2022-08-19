import pickle
import pandas as pd
from tqdm import tqdm

# Le cluster cluster.pkl contient des clusters avec une seule image ainsi que des clusters dupliqués (dûs à la stratégie de multiprocessing).
# Il s'agit ici de le nettoyer, c'est-à-dire :
# - enlever les cluster avec une seule image et pas de bateau.
# ==============================================================================

def supr_small():
    with open("/tf/ship_data/mosaics/clusters/clusters.pkl", "rb") as fp:   # Unpickling
        clusters = pickle.load(fp)

    # transformer les clusters en liste d'images, c'est-à-dire sans coordonnées, et supprimer les clusters d'une longueur strict. inf. à 2.
    clusters_l = []
    for cluster in tqdm(clusters):
        clust_l = [img[0] for img in cluster]
        if len(clust_l) > 1 :
            clusters_l.append(clust_l)

    # on sauvegarde le résultat
    f = open('/tf/ship_data/mosaics/clusters/clusters_clean.pkl', "wb") 
    pickle.dump(clusters_l, f)
    f.close()   

def supr_duplicates():
    with open('/tf/ship_data/mosaics/clusters/clusters_clean.pkl', "rb") as fp:   # Unpickling
        clusters = pickle.load(fp)

    clusters_copy = clusters.copy()
    new_clusters = []
    print('Longueur initalie : '+str(len(clusters_copy)))
    i = 0

    while len(clusters_copy) != 0:
        cluster = clusters_copy.pop()
        set_clust = set(cluster)

        for cluster_c in clusters_copy: # on compare la première image de cluster avec celles des clusters (cluster_c) de cluster_copy
            set_clust_c = set(cluster_c)
            if len(set_clust & set_clust_c) != 0 :
                clusters_copy.remove(cluster_c)
                if len(set_clust_c)>len(set_clust) : # on ne conservera que le plus grand des deux
                    cluster = cluster_c
                    set_clust = set(cluster)
        new_clusters.append(cluster)

        i+=1
        if i%100 == 0:
            print(len(clusters_copy))

    # on sauvegarde le résultat
    f = open('/tf/ship_data/mosaics/clusters/clusters_clean_2.pkl', "wb") 
    pickle.dump(new_clusters, f)
    f.close()
    
if __name__=='__main__':

    with open("/tf/ship_data/mosaics/clusters/clusters_clean_2.pkl", "rb") as fp:   # Unpickling
         clusters = pickle.load(fp)

    print(len(clusters))

    # new_clusters = supr_duplicates(clusters)
    # f = open('/tf/ship_data/mosaics/clusters/clusters_clean3.pkl', "wb") 
    # pickle.dump(new_clusters, f)
    # f.close()

    # with open("/tf/ship_data/mosaics/clusters/clusters_clean3_l.pkl", "rb") as fp:   # Unpickling
    #      clusters = pickle.load(fp)

    # clusters_l = []

    # for cluster in clusters:
    #     cluster_l = [img[0] for img in cluster]
    #     clusters_l.append(cluster_l)

    # f = open('/tf/ship_data/mosaics/clusters/clusters_clean3_l.pkl', "wb") 
    # pickle.dump(clusters_l, f)
    # f.close()

    # data_label = pd.read_csv('/tf/ship_data/train_ship_segmentations_v2.csv')
    # imgs_w_boats = data_label[data_label.EncodedPixels == data_label.EncodedPixels]['ImageId'].unique()

    # i = 0
    # for cluster in clusters :
    #     for 






    # data_label = pd.read_csv('/tf/ship_data/train_ship_segmentations_v2.csv')
    # imgs_w_boats = data_label[data_label.EncodedPixels == data_label.EncodedPixels]['ImageId'].unique()
    # for cluster in tqdm(clusters):
    #     if len(cluster) == 1:
    #         if cluster[0][0] not in imgs_w_boats : 
    #             print(cluster)
    #             print('faux')
    #             break

    

        