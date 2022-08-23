import albumentations as A
import pandas as pd
import cv2
import os
import numpy as np
from tqdm import tqdm

def rle2bbox(rle, shape):
    '''
    adapted from : https://www.kaggle.com/code/eigrad/convert-rle-to-bounding-box-x0-y0-x1-y1/notebook
    '''
    if rle == rle : # test if rle != NaN
        a = np.fromiter(rle.split(), dtype=np.uint)
        a = a.reshape((-1, 2))  # an array of (start, length) pairs
        a[:,0] -= 1  # `start` is 1-indexed
        
        y0 = a[:,0] % shape[0]
        y1 = y0 + a[:,1]
        if np.any(y1 > shape[0]):
            # got `y` overrun, meaning that there are a pixels in mask on 0 and shape[0] position
            y0 = 0
            y1 = shape[0]
        else:
            y0 = np.min(y0)
            y1 = np.max(y1)
        
        x0 = a[:,0] // shape[0]
        x1 = (a[:,0] + a[:,1]) // shape[0]
        x0 = np.min(x0)
        x1 = np.max(x1)
        
        if x1 > shape[1]:
            # just went out of the image dimensions
            raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (
                x1, shape[1]
            ))

        return x0, y0, x1, y1

class Image:

    def __init__(self, img_dir, img_name, mask_dir, path_od_csv):
        self.img = cv2.imread(os.path.join(img_dir,img_name))
        self.mask_name = img_name[:img_name.index('.')]+'_mask'+'.png'
        self.mask = cv2.imread(os.path.join(mask_dir,self.mask_name))
        self.name = img_name
        df = pd.read_csv(path_od_csv)
        self.df = df[df.filename == img_name]
        self.height, self.width, self.channels = self.img.shape
        self.bboxes = [[row['xmin'], row['ymin'], row['xmax'], row['ymax'], index] for index,row in self.df.iterrows()]
        
    def transform(self, k, rot):
        """
        Applique soit pas de modification géométrique(k=0), soit une symétrie selon l'axe des abscisses (k=1), soit une symétrie selon l'axe des ordonnées(k=2).
        Applique également les transformations niveau pixels de flou, de teinte, de luminosité et de constraste ayant toutes pour probabilité 0.5 d'être appliquées à notre objet.
        Applique également une rotation multiple de 90° dans le sens trigo.
        Nomenclature : une image augmentée aura un nom de la forme 567261abb_t_vert_rot90.jpg
        - t : signifie que l'image a été transformée (niveau pixels), c'est à dire qu'au moins une des quatre transformations ci-dessus a été appliquée.
        - vert/ horiz : signique qu'une symétrie selon l'axe des ordonnées (vert) ou selon l'axe des abscisses (horiz) a été appliquée.
        - rot90/rot180/rot270 : signique qu'une rotation de l'angle donné dans le sens trigo a été effectuée.

        :paramp vert_p: float, probabilité d'appliquer une symétrie selon l'axe des abscisses.
        :param hori_p: float, probabilité d'appliquer une symétrie selon l'axe des ordonnées.
        :param k: int, pas de modification géométrique, 1 si symétrie selon l'axe des abscisses, 2 si symétrie selon l'axe des ordonnées.
        :param rot: int, 0 si pas de rotation, 1 si 90°; 2 si 180°, 3 si 270°
        :return: Void.
        """
        if k == 0 : 
            vert_p, hori_p = 0, 0
        elif k == 1 :
            vert_p, hori_p = 1, 0
        elif k == 2 : 
            vert_p, hori_p = 0, 1
        else : 
            print('Choisir une valeur de k dans {0, 1, 2}')
            return

        if rot == 0 :
            r, p_r = 0, 0
        elif rot == 1 :
            r, p_r = 90, 1
        elif rot == 2 :
            r, p_r = 180, 1
        elif rot == 3 : 
            r, p_r = 270, 1
        else :
            print('Choisir une valeur de rot dans {0, 1, 2, 3}')
            return

        # transformation using albumentations
        transform = A.Compose(
            [A.VerticalFlip(p=vert_p),
            A.HorizontalFlip(p=hori_p),
            A.Rotate(limit=(r,r), p=p_r),
            A.ColorJitter(),
            A.Blur(blur_limit=5),
            A.RandomBrightnessContrast()], 
            bbox_params=A.BboxParams(format='pascal_voc'))
        
        # appliquer à l'image
        transformed = transform(image=self.img,mask=self.mask,bboxes=self.bboxes)

        # changer les attribut de l'objet
        self.img = transformed['image']
        self.mask = transformed['mask']
        self.bboxes = transformed['bboxes']
        if k == 1 : 
            self.name = self.name[:self.name.rfind('.')]+'_t_horiz.jpg'
            self.mask_name = self.mask_name[:self.mask_name.rfind('.')]+'_t_horiz.png'
        elif k == 2 : 
            self.name = self.name[:self.name.rfind('.')]+'_t_vert.jpg'
            self.mask_name = self.mask_name[:self.mask_name.rfind('.')]+'_t_vert.png'
        else :
            self.name = self.name[:self.name.rfind('.')]+'_t.jpg'
            self.mask_name = self.mask_name[:self.mask_name.rfind('.')]+'_t.png'

        if rot == 1 : 
            self.name = self.name[:self.name.rfind('.')]+'_rot90.jpg'
            self.mask_name = self.mask_name[:self.mask_name.rfind('.')]+'_rot90.png'
        elif rot == 2 : 
            self.name = self.name[:self.name.rfind('.')]+'_rot180.jpg'
            self.mask_name = self.mask_name[:self.mask_name.rfind('.')]+'_rot180.png'
        elif rot == 3 :
            self.name = self.name[:self.name.rfind('.')]+'_rot270.jpg'
            self.mask_name = self.mask_name[:self.mask_name.rfind('.')]+'_rot270.png'
        for (index, row), bbox in zip(self.df.iterrows(), self.bboxes):
            self.df.loc[index,'xmin'], self.df.loc[index,'ymin'], self.df.loc[index,'xmax'], self.df.loc[index,'ymax'] = bbox[:4]
            self.df.loc[index,'filename'] = self.name

    def save(self, saving_dir, mask_saving_dir=None): # préciser mask_dir=*qql_chose* si l'on souhaite que le masque de l'image soit également enregistré
        cv2.imwrite(os.path.join(saving_dir, self.name), self.img)
        if mask_saving_dir is not None:
            cv2.imwrite(os.path.join(mask_saving_dir, self.mask_name), self.mask)

def augment_img(img_name, img_dir, mask_dir, img_saving_dir, mask_saving_dir, path_od_csv):
    """
    Procède à la creation de 7 images 'augmentées' à partir d'une image de départ, des masques corresponds ainsi que  
    Ces 7 images correspondent aux 7 possibles transformations de type symétrie axiale ou rotation faisables,
    auxquelles on ajoute différentes modifications de flou, de teinte, de luminosité et de constraste ayant toutes pour probabilité 0.5 d'être appliquées.
    Renvoie le nouveau dataframe formé au format pascal VOC.

    :param img_name: str, nom de l'image.
    :param img_dir: str, répertoire où se trouve l'image.
    :param mask_dir: str, répertoire où se trouve le masque de l'image.
    :param img_saving_dir: str, répertoire où seront enregistrées ces 7 images.
    :param mask_saving_dir: str, répertoire où seront enregistrées les masques correspondants.
    :param path_od_csv: str, chemin du CSV original au format pascal VOC.
    :return: df, nouveau dataframe formé par les 7 images, au format pascal VOC.
    """

    # On peut former 7 nouvelles images à partir de transformation de type symétrie axiale et rotation

    # 1 - vertical flip 
    img_obj1 = Image(img_dir, img_name, mask_dir, path_od_csv)
    img_obj1.transform(2,0)
    img_obj1.save(img_saving_dir, mask_saving_dir)

    # 2 - vertical flip + 90° rotation
    img_obj2 = Image(img_dir, img_name, mask_dir, path_od_csv)
    img_obj2.transform(2,1)
    img_obj2.save(img_saving_dir, mask_saving_dir)

    # 3 - horizontal flip 
    img_obj3 = Image(img_dir, img_name, mask_dir, path_od_csv)
    img_obj3.transform(1,0)
    img_obj3.save(img_saving_dir, mask_saving_dir)

    # 4 - horizontal flip + 90° rotation
    img_obj4 = Image(img_dir, img_name, mask_dir, path_od_csv)
    img_obj4.transform(1,1)
    img_obj4.save(img_saving_dir, mask_saving_dir)

    # 5 - just 90° rotation
    img_obj5 = Image(img_dir, img_name, mask_dir, path_od_csv)
    img_obj5.transform(0,1)
    img_obj5.save(img_saving_dir, mask_saving_dir)

    # 6 - just 180° rotation
    img_obj6 = Image(img_dir, img_name, mask_dir, path_od_csv)
    img_obj6.transform(0,2)
    img_obj6.save(img_saving_dir, mask_saving_dir)

    # 7 - just 270° rotation
    img_obj7 =  Image(img_dir, img_name, mask_dir, path_od_csv)
    img_obj7.transform(0,3)
    img_obj7.save(img_saving_dir, mask_saving_dir)

    # return the new dataframe with new bbox coordinates
    return pd.concat([img_obj1.df, img_obj2.df, img_obj3.df, img_obj4.df, img_obj5.df, img_obj6.df, img_obj7.df])

def augment(img_l, img_dir, img_saving_dir, mask_dir, mask_saving_dir, path_od_csv, path_new_od_csv):
    """
    Effectue l'augmentation de donnée décrite ci-dessus (images et masques) et sauvegarde le csv correspondant.

    :param img_l: list, liste des noms d'images à augmenter.
    :param img_dir: str, répertoire où se trouvent les images.
    :param img_saving_dir: str, répertoire où seront enregistrées les nouvelles images.
    :param mask_dir: str, répertoire où se trouvent les masques des images.
    :param mask_saving_dir: str, répertoire où seront enregistrées les masques correspondants.
    :param path_od_csv: str, chemin du CSV au format pascal VOC.
    :param path_new_od_csv: str, chemin où sera enregistré le nouveau csv.
    :return: Void.
    """
    #########
    # Sur certaines image (une dizaine), les valeurs de xmin et xmax / ymin et ymax sont les mêmes, ce qui génère des erreurs
    # Il faut donc les supprimer du dataframe de départ les lignes correspondantes
    df= pd.read_csv(path_od_csv)
    for index, row in df.iterrows():
        if row['xmax'] == row['xmin'] or row['ymax'] == row['ymin']:
            df = df.drop(index)

    df.to_csv(path_od_csv, index=False)
    #########
    
    df_list = []

    for img_name in tqdm(img_l):
        df = augment_img(img_name, img_dir, mask_dir, img_saving_dir, mask_saving_dir, path_od_csv)
        df_list.append(df)
    
    pd.concat(df_list, ignore_index=True).to_csv(path_new_od_csv, index=False)

if __name__ == '__main__':

    path_train_csv = '/tf/ship_data/annotations/70_80/train_70_80.csv'
    df_train = pd.read_csv(path_train_csv)
    train_img_l = df_train[df_train.xmax == df_train.xmax]['filename'].unique()

    img_l = ['00021ddc3.jpg']
    img_dir = '/tf/ship_data/train_v2'
    mask_dir = '/tf/ship_data/masks_only_one_image'
    img_saving_dir = '/tf/test/results'
    mask_saving_dir = '/tf/test/masks'
    path_od_csv = '/tf/ship_detect_tl/data_parsing/CSV/train_ship_segmentations_OD.csv'
    path_new_od_csv = '/tf/test/augmented_data_OD.csv'

    #  augment(img_l, img_dir, img_saving_dir, mask_dir, mask_saving_dir, path_od_csv, path_new_od_csv)



        





