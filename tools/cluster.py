import cv2
import os
import time
import numpy as np
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count

T = 15

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
    img_list_copy = img_list.copy()
    img_list_copy.remove(img_name)

    to_explore = [(img_name,(0,0))]

    while len(to_explore) != 0  :

        # print('reste à explorer : ' + str(len(to_explore)))
        # print('dans un liste de : '+str(len(img_list_copy)))

        img_spot_name, coords = to_explore.pop()

        # print("exploring at coords : "+str(coords))
        # print('\n')

        for dir in ['N', 'S', 'O', 'E'] :
            if exists_neighbor(img_spot_name, dir, img_list_copy)[0]:
                neighbor_name = exists_neighbor(img_spot_name, dir, img_list_copy)[1]
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
                img_list_copy.remove(neighbor_name)

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
        print(i,j)
        print(el[0])
        img = cv2.imread('/tf/ship_data/train_v2/' + el[0])
        for x in range(3):
            for y in range(3):
                crop = img[x*256: (x+1)*256, y*256: (y+1)*256]
                img_tile[j+x][i+y] = crop

    mosaic = cv2.vconcat([cv2.hconcat(list_h) for list_h in img_tile]) 
    
    cv2.imwrite('/tf/mosaic/'+'mosaic.png',mosaic)

def main():

    img_list = os.listdir('/tf/ship_data/train_v2')[:10000]
    all_clusters = []

    print(f'Starting computations on {cpu_count()} cores')

    while len(img_list) != 0 :

        print("Nombre d'images restantes à traiter : "+str(len(img_list)))

        start = timer()

        # On prend autant d'images que de coeurs 
        img_spot = np.random.choice(img_list, cpu_count())

        # On crée l'itérable qui va être distribué aux workers
        values = [(img_name, img_list) for img_name in img_spot]

        with Pool() as pool:
            clusters = pool.starmap(expand_cluster, values) # clusters est la liste des clusters trouvés par expand_cluster pour chaque worker
        all_clusters.append(clusters)

        # On enlève les images qui appartiennent maintenant à un cluster à la liste d'images de depart
        for cluster in clusters :
            for el in cluster:
                img_name = el[0]
                try:
                    img_list.remove(img_name)
                except ValueError:
                    pass 

        end = timer()
        print(f'elapsed time: {end - start}')
    
    # On sauvegarde les clusters dans un .txt
    f = open('clusters.txt','w')
    for run in all_clusters:
        for cluster in run:
            f.write(str(cluster))
        f.write('\n')
    f.close()

if __name__ == '__main__':
    main()
    # f = open('/tf/clusters.txt','r')
