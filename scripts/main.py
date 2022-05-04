from statistics import mode
from preprocess_data import preprocess_data
from load_model import *
import shutil


# variables

path_original_csv = '/tf/ship_data/train_ship_segmentations_v2.csv'
path_images = '/tf/ship_data_lite'
cut_rate = 0.8
model_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz'

# process data
if not os.path.exists('/tf/ship_detect_tl/scripts/test'):
    preprocess_data(path_original_csv, cut_rate, path_images)

# load the model
if not os.path.exists('/tf/pretrained_models'):
    model = Model()
    model.download_model(model_url)
    os.makedirs(os.path.join('custom_models',model.model_name))
    shutil.copyfile(os.path.join(model.cache_dir,'checkpoints',model.model_name,'pipeline.config'), os.path.join('custom_models',model.model_name,'pipeline.config'))
    
# configure config file 