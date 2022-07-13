import tensorflow as tf
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
from object_detection.utils import dataset_util
import random
import pickle
from sizesV2 import draw_distrib
from sizesV2 import sizes_distrib_csv

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

def generate_tf_record(clusters, path_h_csv, resol, image_rate, cut_rate, boat_rate, tfrecord_dir):
    """
    Génère deux fichiers au format tfrecord dans le répertoire tfrecord_dir : deux fichiers train et test 
    nommés selon train_1000_80_70.tfrecord pour 1000 images, dont 80% d'images avec bateau réparties à 70% dans train (et donc 30% dans test).
    
    :param clusters: list, liste des clusters (eux mêmes listes d'images) à partir desquels vont être formées train et test
    :param path_h_csv: str, chemin du csv des hash des bateaux
    :param resol: int, résolution utilisée (cf. sizes_distrib_csv)
    :param image_rate: float, pourcentage de la base de départ. Pour image_rate=1.0, on aura donc au plus 42556 images avec bateau (nombre maximum disponible dans la base)
    :param cut_rate: float, pourcentage d'images utilisées pour former le tfrecord train. Le tfrecord test sera formé avec les images restantes.
    :param boat_rate: float, pourcentage du nombre total d'images qui doit contenir au moins un bateau
    :param tfrecord_dir: str, répertoire où seront créés les tf records
    :return: Void
    """

    if not os.path.exists(tfrecord_dir):
        os.mkdir(tfrecord_dir)

    saving_dir = os.path.join(tfrecord_dir,str(int(image_rate*100))+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))) # subfolder de tfrecord_dir où seront stockés les tfrecords et metadonnées sur les bases créées

    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)

    if os.path.exists(os.path.join(saving_dir,'train_'+str(int(image_rate*100))+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.tfrecord')):
        print('Les tfrecords demandés sont déjà présents au chemin suivant :'+ saving_dir)
        return

    # 1. créer deux dataframes df_train et df_test selon les paramètres : cut rate, boat_rate, image_rate

    df = pd.read_csv(path_to_csv)
    df_train = pd.DataFrame(columns=df.columns.tolist())
    df_test = pd.DataFrame(columns=df.columns.tolist())

        # on crée deux listes de noms d'images : une correspondant à la base d'entrainement et l'autre à la base de test.
        # idée du fonctionnement : on remplie d'abord ces deux listes avec les noms d'images contenant des bateaux, puis avec le noms des images vides. Enfin, on mélange les listes ainsi crées.











    df_scores = pd.read_csv('/tf/ship_data/boats_sizes/boats_sizes_scores_resol6.csv')
    df_sample = df_scores.groupby(['W_score','H_score'], group_keys=False).apply(lambda x: x.sample(frac=image_rate)) # dataframe des images avec bateau

    df_sample_test = df_sample.groupby(['W_score','H_score'], group_keys=False).apply(lambda x: x.sample(frac=1-cut_rate)) # dataframe des images avec bateau et qui iront dans la base test
    sample_boats_names_test = df_sample_test['filename'].unique() # liste des noms des images avec bateau et qui iront dans la base test

    df_sample_train = df_sample[df_sample['filename'].isin(sample_boats_names_test) == False] # dataframe des images avec bateau et qui iront dans la base test
    sample_boats_names_train = df_sample_train['filename'].unique() # liste des noms des images avec bateau et qui iront dans la base train
    
    l_train = sample_boats_names_train.tolist().copy()
    l_test = sample_boats_names_test.tolist().copy()

        # Puis, on complète avec des images sans bateau :
    nb_image_boat = len(sample_boats_names_test) + len(sample_boats_names_train)
    df_no_ship = df[df.xmax != df.xmax] # dataframe des images sans bateau
    j = 0
    for image_name in df_no_ship['filename'].unique()[:int(nb_image_boat/boat_rate-nb_image_boat)]:
        if j < int(len(sample_boats_names_train)/boat_rate-len(sample_boats_names_train)):
            l_train.append(image_name)
            j+=1
        else :
            l_test.append(image_name)


















    
        # On mélange les listes ainsi obtenues :
    random.shuffle(l_train)
    random.shuffle(l_test)

        # Enfin, on convertit ces listes de noms d'images en dataframes format pascal VOC avec les bboxs correspondantes :
    print('Création des dataframes train et test...')
    for image_name in tqdm(l_train):
        image_df = df[df.filename == image_name]
        df_train = pd.concat([df_train,image_df], ignore_index = True)
    for image_name in tqdm(l_test):
        image_df = df[df.filename == image_name]
        df_test = pd.concat([df_test,image_df], ignore_index = True)

        # On garde une trâce au format CSV des images présentes dans les bases de données :
    train_csv_name = 'train_'+str(int(image_rate*100))+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.csv'
    test_csv_name = 'test_'+str(int(image_rate*100))+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.csv'
    df_train.to_csv(os.path.join(saving_dir, train_csv_name), index=False)
    df_test.to_csv(os.path.join(saving_dir, test_csv_name), index=False)

        # On représente aussi les distributions de tailles pour s'assurer que l'échantillon test est bien représentatif de la base d'entrainement
    print('Création de la distribution de tailles de bboxs dans la base train (pdf)')
    draw_distrib(os.path.join(saving_dir, train_csv_name), 384, os.path.join(saving_dir,'train'))
    print('Création de la distribution de tailles de bboxs dans la base test (pdf)')
    draw_distrib(os.path.join(saving_dir, test_csv_name), 384, os.path.join(saving_dir,'test'))

    # 2. utiliser ce deux dataframes pour créer train.tfrecord et test.tfrecord à l'emplacement tfrecord_dir

        # tfrecord train

    filename_train = 'train_'+str(int(image_rate*100))+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.tfrecord'

    writer_train = tf.io.TFRecordWriter(os.path.join(saving_dir,filename_train))
    list_images_names_train = df_train['filename'].unique()

    print('Création du tfrecord train...')
    for file_name in tqdm(list_images_names_train):
        tf_example = create_tf_example(file_name, path_images, df_train)
        writer_train.write(tf_example.SerializeToString())

    writer_train.close()

        # tfrecord test

    filename_test = 'test_'+str(int(image_rate*100))+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.tfrecord'

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

    path_images = '/tf/ship_data/train_v2'
    path_to_csv = '/tf/ship_data/train_ship_segmentations_OD.csv'
    image_rate = 0.01 # taux d'images avec bateau de la base initiale, sachant qu'au maximum on peut prendre 42556 images avec bateau pour image_rate=1.
    boat_rate = 0.8 # taux d'images contenant au moins un bateau
    cut_rate = 0.9 # taux d'images (par rapport à nb_images) utilisées pour train. Le reste sera utilisé pour test.
    tfrecord_dir = '/tf/tests' # répertoire où train et test seront créés
    generate_tf_record(path_images, path_to_csv, image_rate, cut_rate, boat_rate, tfrecord_dir)

    with open("/tf/ship_data/find_duplicates/reassemble_cluster/cluster_reassembled.pkl", "rb") as fp:   # Unpickling
         clusters = pickle.load(fp)
    path_h_csv = '/tf/ship_data/find_duplicates/hash/boats_hashV2.csv'
    resol = 8 
    