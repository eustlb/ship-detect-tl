import pickle
import pandas as pd
from tqdm import tqdm

def big_clust(cluster):
    """
    Prends un cluster (trouvé par la méthode réseau) et renvoie l'index du cluster 
    (parmis ceux trouvés par la méthode mosaic, liste de noms d'images aussi) le plus 
    grand qui contient au moins une image de ce cluster.
    S'il n'y en a pas, renvoie -1.

    :param cluster: list, liste d'images correspondant à un cluster
    :return: l'index du grand cluster, -1 s'il n'y en a pas.
    """
    with open("/tf/ship_data/find_duplicates/mosaics/clusters/clusters_clean.pkl", "rb") as fp:   # Unpickling
        mosaic_clusters = pickle.load(fp)
    
    set_clust = set(cluster)
    big_clust = []

    for cluster_c in mosaic_clusters:
        set_clust_c = set(cluster_c)
        if len(set_clust & set_clust_c) != 0 and len(cluster_c)>len(big_clust) :
            big_clust = cluster_c
    
    if len(big_clust) != 0:
        return mosaic_clusters.index(big_clust)
    else : 
        return -1

def big_clust_csv():
    """
    Crée le csv qui associe à chaque cluster trouvé par la méthode des hashs son 'big_clust'.
    """
    df_cluster_h = pd.read_csv('/tf/ship_data/find_duplicates/hash/clusters.csv')
    big_clust_dict = {'ClusterId':[], 'Images':[], 'BigClustIndex':[]}

    for index, row in tqdm(df_cluster_h.iterrows(), total=df_cluster_h.shape[0]):
        big_clust_dict['ClusterId'].append(row['ClusterId'])
        big_clust_dict['Images'].append(row['Images'])
        cluster = row['Images'].split(' ')
        big_clust_dict['BigClustIndex'].append(big_clust(cluster))

    df = pd.DataFrame.from_dict(big_clust_dict)
    pd.DataFrame.to_csv(df, '/tf/ship_data/find_duplicates/reassemble_cluster/big_clust.csv')

def reassemble_clusters():
    """
    Si deux clusters trouvés par la méthode réseau possèdent le même "big_clust",
    c'est-à-dire qu'ils appartiennent à la même image satellite de départ,
    alors ils doivent être rassemblés en un même cluster.
    """
    df_cluster_h = pd.read_csv('/tf/ship_data/find_duplicates/reassemble_cluster/big_clust.csv')
    index_l = df_cluster_h['BigClustIndex'].unique().tolist()
    index_l.remove(-1)

    new_clusters = []
    for index in tqdm(index_l):
        cluster = []
        for clust in  df_cluster_h[df_cluster_h.BigClustIndex == index]['Images']:
            clust_l = clust.split(' ')
            for img_name in clust_l:
                if img_name not in cluster:
                    cluster.append(img_name)
        new_clusters.append(cluster)

    # on sauvegarde le résultat
    f = open('/tf/ship_data/find_duplicates/reassemble_cluster/cluster_reassembled.pkl', "wb") 
    pickle.dump(new_clusters, f)
    f.close()

def cluster_csv(clusters, path_h_csv, path_new_csv):
    """
    Crée un csv qui associe a chaque cluster des clusters reassemblés, un identifiant (entier), les images contenues et les bateaux contenus.

    :param clusters: list, liste des clusters reassemblés
    :param path_h_csv: str, chemin du csv des BoatHash.
    :param path_new_csv: str, chemin où sera sauvegardé ce nouveau csv.
    """

    df_h = pd.read_csv(path_h_csv)

    clusters_dict = {'ClusterId':[], 'Images': [], 'BoatHash': []}
    
    for i in tqdm(range(len(clusters))):
        clusters_dict['ClusterId'].append(i)
        clusters_dict['Images'].append(' '.join(clusters[i]))
        boats_h_l = []
        for img_name in clusters[i]:
            for boat_h in [boat_h for boat_h in df_h[df_h.ImageId == img_name]['BoatHash']]:
                if boat_h not in boats_h_l:
                    boats_h_l.append(boat_h)
        clusters_dict['BoatHash'].append(' '.join([str(boat_h) for boat_h in boats_h_l]))
    
    df = pd.DataFrame.from_dict(clusters_dict)
    pd.DataFrame.to_csv(df, path_new_csv)   

if __name__ == "__main__":
    with open("/tf/ship_data/find_duplicates/reassemble_cluster/cluster_reassembled.pkl", "rb") as fp:   # Unpickling
         clusters = pickle.load(fp)

    path_h_csv = '/tf/ship_data/find_duplicates/hash/boats_hash.csv'
    path_new_csv = '/tf/ship_data/find_duplicates/hash/clusters.csv'

    cluster_csv(clusters, path_h_csv, path_new_csv)

    
    