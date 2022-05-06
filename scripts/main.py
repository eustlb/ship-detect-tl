from statistics import mode
from generate_tfrecord import generate_tf_record
from load_model import *
import shutil

# Création des tfrecord
path_images = '/tf/ship_data/train_v2'
path_to_csv = '/tf/ship_data/train_ship_segmentations_OD.csv'
nb_images = 53000
boat_rate = 0.9
cut_rate = 0.7
tfrecord_dir = '/tf/annotations'

if not os.path.exists('/tf/annotations'):
    os.mkdir('/tf/annotations')
generate_tf_record(path_images, path_to_csv, nb_images, cut_rate, boat_rate, tfrecord_dir)

# Chargement du modèle
model_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz'

if not os.path.exists('/tf/pretrained_models'):
    model = Model()
    model.download_model(model_url)
    os.makedirs(os.path.join('custom_models',model.model_name))
    shutil.copyfile(os.path.join(model.cache_dir,'checkpoints',model.model_name,'pipeline.config'), os.path.join('custom_models',model.model_name,'pipeline.config'))
    
