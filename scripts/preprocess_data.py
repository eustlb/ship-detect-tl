import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import cv2
import os
import io
from PIL import Image
from object_detection.utils import dataset_util


def rle2bbox(rle, shape):
    '''
    taken from : https://www.kaggle.com/code/eigrad/convert-rle-to-bounding-box-x0-y0-x1-y1/notebook
    '''
    
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

def generate_od_csv(dataframe, path_images, path_new_csv):
    '''
    Generates a new csv according to Tensorflow object detection format (Pascal VOC) from a dataframe build according to the structure of the original CSV (from kaggle dataset)
    '''

    df = dataframe
    H = 768
    W = 768

    image_names = df['ImageId'].unique() # list of image names
    od_dict = {'filename':[], 'width': [], 'height': [], 'class': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}

    i=0 # will be printed to show progression
    for image in image_names:
        print(i)
        for rle in df.loc[df['ImageId'] == image]['EncodedPixels']:
            img = cv2.imread(os.path.join(path_images, image))
            if rle == rle : # testing here if rle is not a 'NaN'
                xmin, ymin, xmax, ymax = rle2bbox(rle, (H, W))
                od_dict['class'].append('ship')
            else :
                xmin, ymin, xmax, ymax = '', '', '', ''
                od_dict['class'].append('')
            od_dict['filename'].append(image)
            od_dict['width'].append(W)
            od_dict['height'].append(H)
            od_dict['xmin'].append(xmin)
            od_dict['ymin'].append(ymin)
            od_dict['xmax'].append(xmax)
            od_dict['ymax'].append(ymax)
        i+=1

    df_od_train= pd.DataFrame(od_dict)
    df_od_train.to_csv(path_new_csv, index=False)

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

def generate_tf_record(path_images, path_to_csv, path_to_tfrecord):

    writer = tf.io.TFRecordWriter(path_to_tfrecord)

    labels_df = pd.read_csv(path_to_csv)
    list_images_names = labels_df['filename'].unique()
    i = 1
    for file_name in list_images_names:
        print(i)
        tf_example = create_tf_example(file_name, path_images, labels_df)
        writer.write(tf_example.SerializeToString())
        i+=1

    writer.close()

def preprocess_data(path_original_csv, cut_rate, path_images):
    '''
    Parsing des images dans deux répertoires : 'train' et 'test' 
    création des csv correspondants (cut_rate = 0.8 -> 80% train 20 % test) au format TF object detection CSV (Pascal VOC)
    '''

    dir_path = os.path.dirname(os.path.realpath(__file__))

    df = pd.read_csv(path_original_csv)

    dict_train = {'ImageId':[], 'EncodedPixels':[]}
    dict_test = {'ImageId':[], 'EncodedPixels':[]}

    liste_nom_images = os.listdir(path_images)
    print(liste_nom_images)

    if not os.path.exists(dir_path +'/train'):
        os.makedirs(dir_path +'/train')

    if not os.path.exists(dir_path +'/test'):
        os.makedirs(dir_path +'/test')

    for image_name in liste_nom_images:

        i = liste_nom_images.index(image_name)
        print(i)
    
        src = os.path.join(path_images, image_name) # image source
        label = list(df.loc[df['ImageId'] == image_name]['EncodedPixels']) # liste des labels de cette image 

        if i<=int(cut_rate*len(liste_nom_images))-1: # les cut_rate (%) premières images vont dans 'train', les autres dans 'test'
            dest = dir_path +'/train'
            for l in label:
                dict_train['ImageId'].append(image_name)
                dict_train['EncodedPixels'].append(l)
        else :
            dest = dir_path +'/test'
            for l in label:
                dict_test['ImageId'].append(image_name)
                dict_test['EncodedPixels'].append(l)
        
        shutil.move(src, dest) # on déplace l'image
    
    df_train= pd.DataFrame(dict_train)
    print(df_train.head())
    df_test= pd.DataFrame(dict_test)

    generate_od_csv(df_train, dir_path +'/train', dir_path + '/train/train_labels.csv')
    generate_od_csv(df_test, dir_path +'/test', dir_path + '/test/test_labels.csv')

    if not os.path.exists(dir_path + '/annotations'):
        os.makedirs(dir_path + '/annotations')

    generate_tf_record(dir_path + '/train',dir_path + '/train/train_labels.csv', dir_path + '/annotations/train.tfrecord')
    generate_tf_record(dir_path + '/test',dir_path + '/test/test_labels.csv', dir_path + '/annotations/test.tfrecord')