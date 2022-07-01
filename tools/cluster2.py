from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import os
import pickle
import time
import numpy as np
from tqdm import tqdm

T = 5

def find_duplicate(img_name, crop_number1, crop_number2, img_list):
    """
    crop_number1 : numero du crop qui sera regardé dans l'image de départ
    crop_number2 : numero du crop qui sera regardé pour les images de img_list et donc comparé avec le crop de l'image de départ
    """
    rep = (False,'Pas de voisin')
    crop1 = cv2.imread('/tf/imgs_crops/'+img_name[:img_name.index('.')]+'_'+str(crop_number1)+'_'+'.png')
    for img_name_2 in img_list:
        crop2 = cv2.imread('/tf/imgs_crops/'+img_name_2[:img_name_2.index('.')]+'_'+str(crop_number2)+'_'+'.png')
        dist = cv2.norm(crop1 - crop2, cv2.NORM_L2)
        if dist < T :
            return (True,img_name_2,dist)
    return rep

def exists_neighbor(img_name, direction, img_list):
    """
    Prend un nom d'image et un direction et renvoie si il y a un voisin dans la direction donnée dans la liste img_list
    Renvoie (True, 'non_du_voisin') ou (False,'Pas de voisin')
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

    # img_list = os.listdir('/tf/ship_data/train_v2')
    # img_list = ['73c34faed.jpg' ,'69aa9f0f4.jpg' ,'acecdc9ad.jpg' ,'fc1d0f5f5.jpg', '8020e260c.jpg', '88c910ecb.jpg', 'a7bcc4634.jpg' ,'58e2d0fb8.jpg' ,'220df0d70.jpg'] + ['ec4167884.jpg', '7720cc64b.jpg', '31f0f5cd2.jpg', '4e3393ed5.jpg', '34cd21098.jpg', '52cbb54fc.jpg','09e29c7f7.jpg','20d6219ad.jpg','bb59bcb41.jpg', '34cd21098.jpg']

    # main(img_list)

    with open("/tf/clusters/clusters_0.pkl", "rb") as fp:   # Unpickling
        clusters = pickle.load(fp)

    for cluster in tqdm(clusters):
        if len(cluster) > 1 : 
            rebuild_mosaic(cluster)






    
 
    

    



