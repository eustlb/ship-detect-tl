from sys import path
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import pickle
from matplotlib import pyplot as plt

def draw_distrib(boats_h_l, path_h_csv, nb_bar, saving_path):
    """
    Crée un diagramme à barres (format pdf) de la distribution des largeurs et hateurs des bateaux présents dans boats_h_l.

    :param boats_h_l: list, liste des hashs des bateaux dont on veut avoir la distribution en taille. 
    :param path_h_csv: str, path du csv qui lit les hash des bateaux à leurs tailles
    :param nb_bar: int, nombre de barres du diagramme.
    :param saving_dir: str, path où sera enregistré le pdf.
    :return: Void.
    """

    df_h = pd.read_csv(path_h_csv)
    sizes = pd.DataFrame(columns = ['filename','W' , 'H'])

    bar_width = 768 // nb_bar
    counts_w = np.zeros(nb_bar)
    counts_h = np.zeros(nb_bar)

    for boat_h in tqdm(boats_h_l):
        w = df_h[df_h.BoatHash == boat_h]['W'].unique()[0]
        h = df_h[df_h.BoatHash == boat_h]['H'].unique()[0]
        counts_w[int(w//bar_width)]+=1
        counts_h[int(h//bar_width)]+=1
        
    x = np.linspace(bar_width/2,768+bar_width/2,nb_bar,endpoint=False)
    
    fig, ax = plt.subplots(2, 1, figsize=(10,12))
    plt.subplots_adjust(hspace=0.25)
    
    ax[0].set_xticks(np.arange(0, 776, 25))
    ax[0].tick_params(axis='x', labelrotation=45)
    ax[0].set_xlabel('taille en pixels')
    ax[0].set_ylabel("Nombre de bateaux")
    ax[0].set_title('Distribution de la largeur des bateaux sur les images')
    ax[0].bar(x, counts_w, bar_width, color='#6495ed' )
    
    ax[1].set_xticks(np.arange(0, 776, 25))
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].set_xlabel('taille en pixels')
    ax[1].set_ylabel("Nombre de bateaux")
    ax[1].set_title('Distribution de la hauteur des bateaux sur les images')
    ax[1].bar(x, counts_h, bar_width, color='#6495ed' )
    
    plt.savefig(saving_path)

def clusters_sizes_csv(path_clust_csv, path_h_csv, path_new_csv):
    """
    Crée un csv avec pour chaque cluster, identifié par un entier, la largeur moyenne (W_mean), hauteur moyenne (H_mean) et le nombre de bateau présents dans le cluster.

    :param path_clust_csv: str, chemin du csv des clusters
    :param path_h_csv: str, chemin du csv contenant les BoatHash
    :param path_new_csv: str, chemin du csv qui va être ainsi créé.
    :return: Void.
    """

    df_clust = pd.read_csv(path_clust_csv)
    df_h = pd.read_csv(path_h_csv)

    clust_dict = {'ClusterId': [], 'W_mean':[], 'H_mean': [], 'n_boats': [], 'Images': [], 'BoatHash':[]}

    for index, row in tqdm(df_clust.iterrows(), total=df_clust.shape[0]):
        clust_dict['ClusterId'].append(row['ClusterId'])
        boats_h_l = [int(h) for h in row['BoatHash'].split(' ')]
        W_l = [df_h[df_h.BoatHash == boat_h]['W'].unique()[0] for boat_h in boats_h_l]
        H_l = [df_h[df_h.BoatHash == boat_h]['H'].unique()[0] for boat_h in boats_h_l]
        clust_dict['W_mean'].append(sum(W_l)/len(W_l))
        clust_dict['H_mean'].append(sum(H_l)/len(H_l))
        clust_dict['n_boats'].append(len(boats_h_l))
        clust_dict['Images'].append(row['Images'])
        clust_dict['BoatHash'].append(row['BoatHash'])
    
    df = pd.DataFrame.from_dict(clust_dict)
    pd.DataFrame.to_csv(df, path_new_csv) 

if __name__ == '__main__' :

    # 1.
    path_clust_csv = '/tf/ship_data/find_duplicates/hash/clustersV2.csv'
    path_h_csv = '/tf/ship_data/find_duplicates/hash/boats_hashV2.csv'
    path_new_csv = '/tf/ship_data/find_duplicates/reassemble_cluster/clusters_sizes.csv'
    clusters_sizes_csv(path_clust_csv, path_h_csv, path_new_csv)

    # # 2.
    # path_h_csv = '/tf/ship_data/find_duplicates/hash/boats_hashV2.csv'
    # nb_bar = 192
    # boats_h_l = pd.read_csv(path_h_csv)['BoatHash'].unique()
    # print(boats_h_l)
    # saving_path = '/tf/ship_data/boat_info/boats_size_all'+'_'+str(nb_bar)+'.pdf'
    # draw_distrib(boats_h_l, path_h_csv, nb_bar, saving_path)






