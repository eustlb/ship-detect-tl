import cv2
import os
import time
import numpy as np
from tqdm import tqdm

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
    dists = []
    img_list.remove(img_name)

    to_explore = [(img_name,(0,0))]

    while len(to_explore) != 0  :

        print('reste à explorer : ' + str(len(to_explore)))
        print('dans un liste de : '+str(len(img_list)))

        img_spot_name, coords = to_explore.pop()

        print("exploring at coords : "+str(coords))
        print('\n')

        for dir in ['N', 'S', 'O', 'E'] :
            if exists_neighbor(img_spot_name, dir, img_list)[0]:
                dists.append(exists_neighbor(img_spot_name, dir, img_list)[2])
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

    return cluster, dists

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

    img_list = os.listdir('/tf/ship_data/train_v2')





    img_name = '4e3393ed5.jpg'
    img_list = os.listdir('/tf/ship_data/train_v2')

    start = time.time()

    result = expand_cluster(img_name, img_list)
    cluster = result[0]
    dists = result[1]
    print(cluster)
    print(dists)
    rebuild_mosaic(cluster)

    end = time.time()

    print(end-start)

if __name__ == '__main__':
    main()
    