import shutil
import cv2
import os
import time
from tqdm import tqdm

T = 50

def find_duplicate(img_name, crop_number1, crop_number2, img_list):
    """
    crop_number1 : numero du crop qui sera regardé dans l'image de départ
    crop_number2 : numero du crop qui sera regardé pour les images de img_list et donc comparé avec le crop de l'image de départ
    """
    rep = (False,'Pas de voisin')
    crop1 = cv2.imread('/tf/imgs_crops/'+img_name[:img_name.index('.')]+'_'+str(crop_number1)+'_'+'.png')
    for img_name_2 in tqdm(img_list):
        crop2 = cv2.imread('/tf/imgs_crops/'+img_name_2[:img_name_2.index('.')]+'_'+str(crop_number2)+'_'+'.png')
        dist = cv2.norm(crop1 - crop2, cv2.NORM_L2)
        if dist < T :
            return (True,img_name_2)
    return rep

def exists_neighbor(img_name, direction, img_list):
    """
    Prend un nom d'image et un direction et renvoie si il y a un voisin dans la direction donnée dans la liste img_list
    Renvoie (True, 'non_du_voisin') ou (False,'Pas de voisin')
    """
    if direction == 'N':
        return find_duplicate(img_name, 4, 7, img_list)

    if direction == 'S':
        return find_duplicate(img_name, 4, 1, img_list)

    if direction == 'O':
        return find_duplicate(img_name, 2, 3, img_list)
        
    if direction == 'E':
        return find_duplicate(img_name, 2, 1, img_list)


def expand_cluster(img_name):
    cluster = [(img_name,(0,0))]

def main():
    img_name = 'ab35eb541.jpg'
    img_list = os.listdir('/tf/ship_data/train_v2')

    start = time.time()

    N = exists_neighbor(img_name, 'N', img_list)
    S = exists_neighbor(img_name, 'S', img_list)
    O = exists_neighbor(img_name, 'O', img_list)
    E = exists_neighbor(img_name, 'E', img_list)

    end = time.time()

    print('N : '+str(N)+'\n')
    shutil.copy('/tf/ship_data/train_v2/'+N[1], '/tf/yo/'+'N_'+N[1])
    print('S : '+str(S)+'\n')
    shutil.copy('/tf/ship_data/train_v2/'+S[1], '/tf/yo/'+'S_'+S[1])
    print('O : '+str(O)+'\n')
    shutil.copy('/tf/ship_data/train_v2/'+O[1], '/tf/yo/'+'O_'+O[1])
    print('E : '+str(E)+'\n')
    shutil.copy('/tf/ship_data/train_v2/'+E[1], '/tf/yo/'+'E_'+E[1])

    print(end-start)

if __name__ == '__main__':
    main()
    