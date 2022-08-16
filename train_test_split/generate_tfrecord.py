import tensorflow as tf
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
from object_detection.utils import dataset_util
import random
import random
from sizes import draw_distrib

def class_text_to_int(row_label):
    """
    Utilisé par le modèle pour faire correspondre au différentes classes un entier. Ici, une seule classe : 'ship'

    :param row_label: str, nom du label. Ici, ça ne pourra être que 'ship'
    :return: int, renvoie l'entier correspondant à la classe row_label
    """
    if row_label == 'ship':
        return 1

def create_tf_example(file_name, path_to_images, labels_df):
    """
    Adaptated from: https://github.com/tensorflow/models/blob/84c0e81fe9683dbdd5ee6b088fa756302f60dc25/research/object_detection/g3doc/using_your_own_dataset.md.
    """
    filename = file_name.encode('utf8') # Filename of the image. Empty if image is not from file
    
    with tf.io.gfile.GFile(os.path.join(path_to_images, file_name), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)

    image = Image.open(encoded_jpg_io)
    width, height = image.size

    image_format = b'jpg' # b'jpeg' or b'png' 

    xmins = []
    xmaxs = [] 
    ymins = [] 
    ymaxs = [] 
    classes_text = [] 
    classes = [] 

    filename_df = labels_df.loc[labels_df['filename'] == file_name]
    filename_df.reset_index()

    for index, row in filename_df.iterrows():
        if row['xmin'] == row['xmin'] :
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def generate_tf_record(path_od_csv, path_h_csv, path_cluster_csv, cut_rate, boat_rate, tfrecord_dir, only_one = True):
    """
    Génère deux fichiers au format tfrecord dans le répertoire tfrecord_dir : deux fichiers train et test 
    nommés selon train_80_70.tfrecord pour 1000 images, dont 80% d'images avec bateau réparties à 70% dans train (et donc 30% dans test).
    
    :param path_od_csv: str, chemin du csv au format pascal VOC
    :param path_h_csv: str, chemin du csv des hash des bateaux
    :param path_cluster_csv: str, chemin du csv cluster_sizes des clusters, contenant les informations ClusterId,W_mean,H_mean,n_boats,Images
    :param cut_rate: float, pourcentage d'images utilisées pour former le tfrecord train. Le tfrecord test sera formé avec les images restantes.
    :param boat_rate: float, pourcentage du nombre total d'images qui doit contenir au moins un bateau
    :param tfrecord_dir: str, répertoire où seront créés les tf records
    :param only_one: bool, default True pour ne sélectionner qu'une seule des images représentant un bateau.
    :return: Void
    """

    # création des répertoires où seront stockés tf_records et metadonnées
    if not os.path.exists(tfrecord_dir):
        os.mkdir(tfrecord_dir)

    saving_dir = os.path.join(tfrecord_dir,str(int(boat_rate*100))+'_'+str(int(cut_rate*100))) # subfolder de tfrecord_dir où seront stockés les tfrecords et metadonnées sur les bases créées

    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)

    if os.path.exists(os.path.join(saving_dir,'train_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.tfrecord')):
        print('Les tfrecords demandés sont déjà présents au chemin suivant :'+ saving_dir)
        return

    # 1. créer deux dataframes df_train et df_test selon les paramètres : cut rate, boat_rate
    df_h = pd.read_csv(path_h_csv)
    df_od = pd.read_csv(path_od_csv)
    df_train = pd.DataFrame(columns=df_od.columns.tolist())
    df_test = pd.DataFrame(columns=df_od.columns.tolist())

        # on crée deux listes de noms d'images : une correspondant à la base d'entrainement et l'autre à la base de test.
        # idée du fonctionnement : on remplie d'abord ces deux listes avec les noms d'images contenant des bateaux, puis avec le noms des images vides. Enfin, on mélange les listes ainsi crées.

    df_clust = pd.read_csv(path_cluster_csv)
    df_red = df_clust.drop(2) # on supprime le cluster numéro 2 parce qu'il contient 5929 bateaux alors que le 2ème plus gros cluster n'en contient que 126, ce qui complique l'échantillonage stratifié

            # on crée de nouvelles colonnes au dataframe en subdivisant W_mean, H_mean et n_boats en *bins* catégories 
    W_score = pd.cut(df_red['W_mean'], bins=10, labels=list(range(10)))
    H_score = pd.cut(df_red['H_mean'], bins=10, labels=list(range(10)))
    n_boats_score = pd.cut(df_red['n_boats'], bins=5, labels=list(range(5)))
    df_red = df_red.assign(W_score = W_score)
    df_red = df_red.assign(H_score = H_score)
    df_red = df_red.assign(n_boats_score = n_boats_score)

            # sachant qu'on ne considère pas le cluster numéro 2 (5929 bateaux), il faut reconsidérer la fraction cut_rate. Puis on peut prélever un échantillon.
    fraction =  (1-cut_rate)/(1-5929/17538)
    df_sample = df_red.groupby(['W_score', 'H_score', 'n_boats_score'], group_keys=False).apply(lambda x: x.sample(frac=fraction)) 

        # liste des images avec bateau dans les bases train et test 
    df_clust = pd.read_csv(path_cluster_csv)
    l_train = []
    l_test = []
    n_boats_train, n_boats_test = 0, 0
    
    if only_one:
        for index, row in df_clust.iterrows():
            if row['ClusterId'] in df_sample['ClusterId']:
                boats = row['BoatHash'].split(' ')
                # on choisi 1 image représentant le bateau 
                for boat in boats :
                    img = random.sample(list(df_h[df_h.BoatHash == int(boat)]['ImageId']),1)
                    if img[0] not in l_test:
                        l_test += img
                n_boats_test += row['n_boats']
            else :
                boats = row['BoatHash'].split(' ')
                # on choisi 1 image représentant le bateau 
                for boat in boats :
                    img = random.sample(list(df_h[df_h.BoatHash == int(boat)]['ImageId']),1)
                    if img[0] not in l_train:
                        l_train += img
                n_boats_train += row['n_boats']

    else :
        for index, row in df_clust.iterrows():
            if row['ClusterId'] in df_sample['ClusterId']:
                l_test += row['Images'].split(' ')
                n_boats_test += row['n_boats']
            else :
                l_train += row['Images'].split(' ')
                n_boats_train += row['n_boats']

        # Puis, on complète avec des images sans bateau :
    im_no_boats = df_od[~df_od['filename'].isin(df_h['ImageId'].unique())]['filename'].unique()
    nb_image_boat = len(l_test) + len(l_train)
    im_no_boats_sample = random.sample(list(im_no_boats), int(nb_image_boat*(1-boat_rate)/boat_rate))

    j = 0
    n_train_ini = len(l_train)
    for image_name in im_no_boats_sample:
        if j < int(n_train_ini*(1-boat_rate)/boat_rate):
            l_train.append(image_name)
            j+=1
        else :
            l_test.append(image_name)

        # On mélange les listes ainsi obtenues :
    random.shuffle(l_train)
    random.shuffle(l_test)

        # Enfin, on convertit ces listes de noms d'images en dataframes format pascal VOC avec les bboxs correspondantes :
    print('Création des dataframes train et test...')
    # train_dict = {'filename':[],'width':[],'height':[],'class':[],'xmin':[],'ymin':[],'xmax':[],'ymax':[],'rle':[]}
    for image_name in tqdm(l_train):
        image_df = df_od[df_od.filename == image_name]
        df_train = pd.concat([df_train,image_df], ignore_index = True)
    for image_name in tqdm(l_test):
        image_df = df_od[df_od.filename == image_name]
        df_test = pd.concat([df_test,image_df], ignore_index = True)

        # On garde une trâce au format CSV des images présentes dans les bases de données :
    train_csv_name = 'train'+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.csv'
    test_csv_name = 'test'+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.csv'
    df_train.to_csv(os.path.join(saving_dir, train_csv_name), index=False)
    df_test.to_csv(os.path.join(saving_dir, test_csv_name), index=False)

        # On représente aussi les distributions de tailles pour s'assurer que l'échantillon test est bien représentatif de la base d'entrainement
    print('Création de la distribution de tailles de bboxs de la base train (pdf)')
    boats_h_l_train = []
    for im_name in tqdm(l_train):
        boats = [int(el) for el in df_h[df_h.ImageId == im_name]['BoatHash']]
        for boat in boats:
            if boat not in boats_h_l_train:
                boats_h_l_train.append(boat)
    draw_distrib(boats_h_l_train, path_h_csv, 384, os.path.join(saving_dir,'train.pdf'))

    print('Création de la distribution de tailles de bboxs de la base test (pdf)')
    boats_h_l_test = []
    for im_name in tqdm(l_test):
        boats = [int(el) for el in df_h[df_h.ImageId == im_name]['BoatHash']]
        for boat in boats:
            if boat not in boats_h_l_train:
                boats_h_l_test.append(boat)
    draw_distrib(boats_h_l_test, path_h_csv, 384, os.path.join(saving_dir,'test.pdf'))

    # Sauvegarde de ces informations dans un fichier txt
    
    n_im_boats_train = df_train[df_train.xmax == df_train.xmax]['filename'].nunique()
    n_im_no_boat_train = df_train[df_train.xmax != df_train.xmax]['filename'].nunique()
    n_im_boats_test = df_test[df_test.xmax == df_test.xmax]['filename'].nunique()
    n_im_no_boat_test = df_test[df_test.xmax != df_test.xmax]['filename'].nunique()
    
    file = open(os.path.join(saving_dir,"metadata.txt"), "w")
    if only_one:
        file.write(f"Chaque bateau n'est réprésenté que par une image tirée aléatoirement."+"\n")
    else :
        file.write(f"Toutes les images représentant un bateau sont présentes."+"\n")
    file.write(f"Taux de répartion des bateaux entre train et test : {1-n_boats_test/(n_boats_train+n_boats_test)} ({n_boats_train} dans train et {n_boats_test} dans test)."+"\n") 
    file.write(f"Taux d'images avec bateau (train, test): {n_im_boats_train/(n_im_boats_train + n_im_no_boat_train)}, {n_im_boats_test/(n_im_boats_test + n_im_no_boat_test)}"+"\n")
    file.write(f"Nombre d'images total : {n_im_boats_test+n_im_boats_train} (train : {n_im_boats_train}, test : {n_im_boats_test})")
    file.close()

    # 2. utiliser ce deux dataframes pour créer train.tfrecord et test.tfrecord à l'emplacement tfrecord_dir

        # tfrecord train

    filename_train = 'train_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.tfrecord'

    writer_train = tf.io.TFRecordWriter(os.path.join(saving_dir,filename_train))
    list_images_names_train = df_train['filename'].unique()

    print('Création du tfrecord train...')
    for file_name in tqdm(list_images_names_train):
        tf_example = create_tf_example(file_name, path_images, df_train)
        writer_train.write(tf_example.SerializeToString())

    writer_train.close()

        # tfrecord test

    filename_test = 'test_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.tfrecord'

    writer_test = tf.io.TFRecordWriter(os.path.join(saving_dir,filename_test))
    list_images_names_test = df_test['filename'].unique()

    print('Création du tfrecord test...')
    for file_name in tqdm(list_images_names_test):
        tf_example = create_tf_example(file_name, path_images, df_test)
        writer_test.write(tf_example.SerializeToString())

    writer_test.close()

    print('Tfrecords créés avec succès !')
    print('Enregistrés dans le répertoire : '+ saving_dir)

if __name__ == "__main__" :
    path_od_csv = '/tf/ship_data/train_ship_segmentations_OD.csv' 'A REMPLACER'
    path_h_csv = '/tf/ship_detect_tl/CSV/boats_hashV2.csv'
    path_cluster_csv = '/tf/ship_detect_tl/CSV/clusters_sizes.csv'
    path_images = '/tf/ship_data/train_v2'
    boat_rate = 0.7 # taux d'images contenant au moins un bateau
    cut_rate = 0.8 # taux d'images (par rapport à nb_images) utilisées pour train. Le reste sera utilisé pour test.
    tfrecord_dir = '/tf/ship_data/annotations' # répertoire où train et test seront créés
    generate_tf_record(path_od_csv, path_h_csv, path_cluster_csv, cut_rate, boat_rate, tfrecord_dir)