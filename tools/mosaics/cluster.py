from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import os
import pickle
import time
import numpy as np
from tqdm import tqdm

# La base "airbus ship detection challenge" est composée d'images 768*768 extraites à partir d'images satellites avec un pas de 256pix
# Ce projet à pour objetif de reconstruire les différentes images satellites à partir desquelles a été formée cette base.
# ==============================================================================

T = 5 # threeshold variable, will be used to tell if to images are to be considered equal (using there euclidian distance)

def find_duplicate(img_name, crop_number1, crop_number2, img_list):
    """
    Prends une image, deux numéros de crops (cf. build_crops.py) et une liste d'images dans laquelle chercher un voisin, 
    c'est-à-dire une image dont une partie recouvrerait une partie de l'image de départ.
    On cherche ici une image dont le crop_number2 correspondra au crop_number1 de l'image de départ.

    :param crop_number1: int, numero du crop qui sera regardé dans l'image de départ.
    :param crop_number2: int, numero du crop qui sera regardé pour les images de img_list et donc comparé avec le crop de l'image de départ.
    :return rep: tuple, (True, *nom de l'image*) s'il trouve un duplicata, (False, 'Pas de voisin)
    """
    rep = (False,'Pas de voisin')
    crop1 = cv2.imread('/tf/ship_data/find_duplicates/mosaics/imgs_crops/'+img_name[:img_name.index('.')]+'_'+str(crop_number1)+'_'+'.png')
    for img_name_2 in img_list:
        crop2 = cv2.imread('/tf/ship_data/find_duplicates/mosaics/imgs_crops/'+img_name_2[:img_name_2.index('.')]+'_'+str(crop_number2)+'_'+'.png')
        dist = cv2.norm(crop1 - crop2, cv2.NORM_L2)
        if dist < T :
            return (True,img_name_2,dist)
    return rep

def exists_neighbor(img_name, direction, img_list):
    """
    Prend un nom d'image et un direction et renvoie si il y a un voisin dans la direction donnée dans la liste img_list.

    :param img_name: str, nom de l'image de départ
    :param direction: str, doit être dans ('N','S','O','E'). Indique la direction dans laquelle chercher un voisin.
    :param  img_list: liste dans laquelle il faut chercher l'image.
    :return: (True, 'non_du_voisin') ou (False,'Pas de voisin')
    """
    if direction == 'N':
        return find_duplicate(img_name, 5, 8, img_list)

    if direction == 'S':
        return find_duplicate(img_name, 5, 2, img_list)

    if direction == 'O':
        return find_duplicate(img_name, 5, 6, img_list)
        
    if direction == 'E':
        return find_duplicate(img_name, 5, 4, img_list)

def expand_cluster(img_name, img_list):

    cluster = [(img_name,(0,0))]

    try:
        img_list.remove(img_name)
    except ValueError:
        pass

    to_explore = [(img_name,(0,0))]

    while len(to_explore) != 0  :

        # print('reste à explorer : ' + str(len(to_explore)))
        # print('dans un liste de : '+str(len(img_list_copy)))

        img_spot_name, coords = to_explore.pop()

        # print("exploring at coords : "+str(coords))
        # print('\n')

        for dir in ['N', 'S', 'O', 'E'] :
            if exists_neighbor(img_spot_name, dir, img_list)[0]:
                neighbor_name = exists_neighbor(img_spot_name, dir, img_list)[1]
                if dir == 'N':
                    neighbor_coords = (coords[0],coords[1]+1)
                if dir == 'S':
                    neighbor_coords = (coords[0],coords[1]-1)
                if dir == 'O':
                    neighbor_coords = (coords[0]-1,coords[1])
                if dir == 'E':
                    neighbor_coords = (coords[0]+1,coords[1])
                cluster.append((neighbor_name, neighbor_coords))
                to_explore.append((neighbor_name, neighbor_coords))
                img_list.remove(neighbor_name)

    return cluster

def rebuild_mosaic(cluster):

    i_max = max([el[1][0] for el in cluster])
    i_min = min([el[1][0] for el in cluster])
    j_max = max([el[1][1] for el in cluster])
    j_min = min([el[1][1] for el in cluster])

    grey_box = np.full((256, 256, 3), (56,62,66), dtype=np.uint8)

    img_tile = [[grey_box for i in range(i_max-i_min+3)] for j in range (j_max-j_min+3)]

    for el in cluster :
        i,j = el[1][0]-i_min, -(el[1][1]-j_max)
        img = cv2.imread('/tf/ship_data/train_v2/' + el[0])
        for x in range(3):
            for y in range(3):
                crop = img[x*256: (x+1)*256, y*256: (y+1)*256]
                img_tile[j+x][i+y] = crop

    mosaic = cv2.vconcat([cv2.hconcat(list_h) for list_h in img_tile]) 

    for el in cluster:
        if el[1][0]==0 and el[1][1]==0:
            base_name = el[0][:el[0].index('.')]
    
    cv2.imwrite('/tf/mosaic/'+str(base_name)+'_mosaic.png',mosaic)

def main(img_list):

    clusters_list = []
    num_workers = 96

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = [executor.submit(expand_cluster, img_name, img_list) for img_name in img_list[:num_workers]]

    i = 0 # utilisé simplement pour afficher la longueur de la liste d'images tous les 100 clusters faits afin d'avoir un suivi 

    prev_len, j = len(img_list), 0 # utilisé pour faire des sauvegardes du travail effectué tous les 10000 clusters créés

    while len(img_list) != 0:
        for future in futures:
            if future.done():
                
                cluster = future.result()
                clusters_list.append(cluster)

                # supprimer les elements du clusters de la liste d'images
                for el in cluster:
                    img_name = el[0]
                    try:
                        img_list.remove(img_name)
                    except ValueError:
                        pass
                    
                # on relance un nouveau processus
                index = futures.index(future)
                futures.remove(future)
                if len(img_list)>0:
                    img_name = img_list.pop()
                    futures.insert(index,executor.submit(expand_cluster, img_name, img_list))

                i += 1

                if i%100 == 0 :
                    print(len(img_list))  
        
        if prev_len-len(img_list)>10000:
            f = open('/tf/clusters/clusters_'+str(j)+'.pkl', "wb") 
            cluster_list_copy = [cluster for cluster in clusters_list]
            pickle.dump(cluster_list_copy, f)
            f.close()
            j+=1
            prev_len = len(img_list)

    
    # sauvegarde de la liste des clusters sur le disque
    f = open('/tf/clusters/clusters.pkl', "wb") 
    cluster_list = [cluster for cluster in clusters_list]
    pickle.dump(cluster_list, f)
    f.close()

if __name__ == "__main__":
    with open("/tf/ship_data/find_duplicates/reassemble_cluster/cluster_reassembled.pkl", "rb") as fp:   # Unpickling
         clusters = pickle.load(fp)
    
    small_clust = []
    for cluster in clusters:
        if len(cluster)>50:
            small_clust.append(cluster)
    
    for cluster in small_clust[:20]:
        clust = expand_cluster(cluster[0],cluster)
        rebuild_mosaic(clust)






    
 
    

    



