import tensorflow as tf
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
from object_detection.utils import dataset_util

def class_text_to_int(row_label):
    if row_label == 'ship':
        return 1

def create_tf_example(file_name, path_to_images, labels_df):
    # TODO(user): Populate the following variables from your example.
    
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

def generate_tf_record(path_images, path_to_csv, nb_images, cut_rate, boat_rate, tfrecord_dir):
    """
    Génère deux fichiers au format tfrecord dans le répertoire tfrecord_dir : deux fichiers train et test 
    nommés selon train_1000_80_70.tfrecord pour 1000 images, 80% d'images avec bateau réparties 70% dans train.
    path_images : répertoire où sont stockées les images
    path_to_csv : chemin vers le csv qui doit être au format Pascal VOC
    nb_images : nombre total d'images
    cut_rate : pourcentage d'images utilisées pour formerle tfrecord train. Le tfrecord test sera sont formé avec les images restantes
    boat_rate : pourcentage du nombre total d'images qui doit contenir au moins un bateau
    tfrecord_dir : répertoire où seront créés les tf records
    """

    if not os.path.exists(tfrecord_dir):
        os.mkdir(tfrecord_dir)

    if os.path.exists(os.path.join(tfrecord_dir,'train_'+str(nb_images)+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.tfrecord')):
        print('Les tfrecords demandés sont déjà présents au chemin suivant :'+str(tfrecord_dir))
        return 

    # 1. créer deux dataframes df_train et df_test selon les paramètres : cut rate, boat_rate, nb_images

    df = pd.read_csv(path_to_csv)

    df_no_ship = df[df.xmax != df.xmax] # dataframe des images sans bateau
    df_ship = df[df.xmax == df.xmax] # dataframe des images avec bateau

    df_train = pd.DataFrame(columns=df.columns.tolist())
    df_test = pd.DataFrame(columns=df.columns.tolist())

    i, j = 0, 0
    print('Création des dataframes train et test...')
    for image_name in tqdm(df_ship['filename'].unique()[:int(nb_images*boat_rate)]):
        if i < int(cut_rate*int(nb_images*boat_rate)):
            image_df = df_ship[df_ship.filename == image_name]
            df_train = pd.concat([df_train,image_df], ignore_index = True)
            i+=1
        else :
            image_df = df_ship[df_ship.filename == image_name]
            df_test = pd.concat([df_test,image_df], ignore_index = True)

    for image_name in tqdm(df_no_ship['filename'].unique()[:nb_images-int(nb_images*boat_rate)]):
        if j < int(cut_rate*(nb_images-int(nb_images*boat_rate))):
            image_df = df_no_ship[df_no_ship.filename == image_name]
            df_train = pd.concat([df_train,image_df], ignore_index = True)
            j+=1
        else :
            image_df = df_no_ship[df_no_ship.filename == image_name]
            df_test = pd.concat([df_test,image_df], ignore_index = True)

    # utiliser ce deux dataframes pour créer train.tfrecord et test.tfrecord à l'emplacement tfrecord_dir
    # tfrecord train

    filename_train = 'train_'+str(nb_images)+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.tfrecord'

    writer_train = tf.io.TFRecordWriter(os.path.join(tfrecord_dir,filename_train))
    list_images_names_train = df_train['filename'].unique()

    print('Création du tfrecord train...')
    for file_name in tqdm(list_images_names_train):
        tf_example = create_tf_example(file_name, path_images, df_train)
        writer_train.write(tf_example.SerializeToString())

    writer_train.close()

    # tfrecord test

    filename_test = 'test_'+str(nb_images)+'_'+str(int(boat_rate*100))+'_'+str(int(cut_rate*100))+'.tfrecord'

    writer_test = tf.io.TFRecordWriter(os.path.join(tfrecord_dir,filename_test))
    list_images_names_test = df_test['filename'].unique()

    print('Création du tfrecord test...')
    for file_name in tqdm(list_images_names_test):
        tf_example = create_tf_example(file_name, path_images, df_test)
        writer_test.write(tf_example.SerializeToString())

    writer_test.close()

    print('Tfrecords créés avec succès !')
    print('Enregistrés dans le répertoire :'+tfrecord_dir)