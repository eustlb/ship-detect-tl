from distutils.command.build import build
import os
import cv2
from tqdm import tqdm

dir = '/tf/imgs_crops'

def build_crops(img_path):
    """
    Prends le path d'une image et renvoie la liste de ces neuf crops. 
    Un crop est un carré de 256x256pixs, réduit a un carré de 6x6pixs en utilisant l'interpolation inter_area.
    Une image à donc 9 crops car on se déplace avec un pas de 256pix.
    Les crops sont numérotés de 1 à 9 gauche à droite et de base en haut.

    :param img_path: str, path de l'image dont on souhaite construire les crops
    :return crops: liste des 9 crops dans l'ordre de 1 à 9.
    """
    crops = []
    step = 256
    img = cv2.imread(img_path)
    for x_grid in range(0,768,256):
        for y_grid in range(0,768,256):
            crop = img[x_grid: x_grid+step, y_grid: y_grid+step]
            crop = cv2.resize(crop,(6,6),interpolation = cv2.INTER_AREA)
            crops.append(crop)
    return crops

def build_crops(dir, saving_dir):
    """
    Construit les 9 crops de chaque image présente dans dir et les enregistre dans le répertoire saving_dir.

    :param dir: str, répertoire où sont les images.
    :param saving_dir: str, répertoire où seront sauvegardés les crops.
    :return: Void.
    """
    for img_name in tqdm(os.listdir(dir)):
        crops = build_crops(os.path.join(dir,img_name))
        i = 1
        for crop in crops :
            cv2.imwrite(os.path.join(saving_dir,img_name[:img_name.index('.')]+'_'+str(i)+'_'+'.png'),crop)
            i+=1

if __name__ == '__main__':
    dir = '/tf/ship_data/train_v2'
    saving_dir = '/tf/imgs_crops'
    build_crops(dir,saving_dir)
