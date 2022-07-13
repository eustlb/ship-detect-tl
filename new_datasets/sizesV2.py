from sys import path
import pandas as pd
from sklearn import cluster
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
    """

    df_clust = pd.read_csv(path_clust_csv)
    df_h = pd.read_csv(path_h_csv)

    clust_dict = {'ClusterId': [], 'W_mean':[], 'H_mean': [], 'n_boats': [], 'Images': []}

    for index, row in tqdm(df_clust.iterrows(), total=df_clust.shape[0]):
        clust_dict['ClusterId'].append(row['ClusterId'])
        boats_h_l = [int(h) for h in row['BoatHash'].split(' ')]
        W_l = [df_h[df_h.BoatHash == boat_h]['W'].unique()[0] for boat_h in boats_h_l]
        H_l = [df_h[df_h.BoatHash == boat_h]['H'].unique()[0] for boat_h in boats_h_l]
        clust_dict['W_mean'].append(sum(W_l)/len(W_l))
        clust_dict['H_mean'].append(sum(H_l)/len(H_l))
        clust_dict['n_boats'].append(len(boats_h_l))
        clust_dict['Images'].append(row['Images'])
    
    df = pd.DataFrame.from_dict(clust_dict)
    pd.DataFrame.to_csv(df, path_new_csv) 

def sizes_distrib_csv(boats_h_l, path_h_csv, dir_new_csv, resol) :
    """
    create_sizes_csv permet, à partir d'une liste de bateaux et du csv associant à chaque bateau, identifié par son hash, un score de longueur et de largeur selon une certaine résolution.
    Puis, on divise le segment [0; W_max] (resp. [0; H_max]) en 2^(la résolution) segments égaux et on affecte par odre croissant un numéro à chaque segment en partant de 0.
    Par exemple, pour une résolution de 1, une largeur de 124 aura pour score 0, puisque [0; W_max] = [0; 443] donc divisé en 2¹ segments : [0; 221] (auquel appartient 124, d'où son score) et [0; 222].
    Ces grâces à ce csv de scores qu'on va pouvoir extraire un échantillon réprésentatif de la base complète en utilisant l'échantillonnage stratifié.
    """

    W_MAX = 443
    H_MAX = 335

    df_h = pd.read_csv(path_h_csv) # on lit le CSV liant chaque bateau à sa hateur et sa largeur.

    size_dict = {'BoatHash':[], 'W_score':[], 'H_score':[]}

    for boat_h in tqdm(boats_h_l):
        w = df_h[df_h.BoatHash == boat_h]['W'].unique()[0]
        h = df_h[df_h.BoatHash == boat_h]['H'].unique()[0]
        W_score, H_score = int(2**resol*w/W_MAX), int(2**resol*h/H_MAX)
        size_dict['BoatHash'].append(boat_h)
        size_dict['W_score'].append(W_score)
        size_dict['H_score'].append(H_score)
    
    df = pd.DataFrame.from_dict(size_dict)
    pd.DataFrame.to_csv(df, os.path.join(dir_new_csv,'boats_sizes_scores_resol'+str(resol)+'.csv'))  

    print('csv successfully created at : '+ os.path.join(dir_new_csv,'boats_sizes_scores_resol'+str(resol)+'.csv'))

if __name__ == '__main__' :

    # # 1.
    # path_clust_csv = '/tf/ship_data/find_duplicates/hash/clustersV2.csv'
    # path_h_csv = '/tf/ship_data/find_duplicates/hash/boats_hashV2.csv'
    # path_new_csv = '/tf/ship_data/find_duplicates/reassemble_cluster/clusters_sizes.csv'
    # clusters_sizes_csv(path_clust_csv, path_h_csv, path_new_csv)

    # # 2.
    # path_h_csv = '/tf/ship_data/find_duplicates/hash/boats_hashV2.csv'
    # boats_h_l = pd.read_csv(path_h_csv)['BoatHash'].unique()
    # dir_new_csv = '/tf/ship_data/boat_info'
    # resol = 8
    # sizes_distrib_csv(boats_h_l, path_h_csv, dir_new_csv, resol)

    3.
    path_h_csv = '/tf/ship_data/find_duplicates/hash/boats_hashV2.csv'
    nb_bar = 384
    boats_h_l = pd.read_csv(path_h_csv)['BoatHash'].unique()
    saving_path = '/tf/ship_data/boat_info/boats_size_all'+'_'+str(nb_bar)+'.pdf'
    draw_distrib(boats_h_l, path_h_csv, nb_bar, saving_path)






