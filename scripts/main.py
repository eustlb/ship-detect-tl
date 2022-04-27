from preprocess_data import preprocess_data
from load_model import *

# variables

path_original_csv = '/tf/ship_data/train_ship_segmentations_v2.csv'
path_images = '/tf/ship_data_lite'
cut_rate = 0.8
model_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d2_coco17_tpu-32.tar.gz'

# process data
if not os.path.exists('./test'):
    preprocess_data(path_original_csv, cut_rate, path_images)

# load the model
if not os.path.exists('./pretrained_models'):
    model = Model()
    model.download_model(model_url)