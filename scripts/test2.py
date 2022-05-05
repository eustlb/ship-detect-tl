from generate_tfrecord import *

path_images = '/tf/ship_data/train_v2'
path_to_csv = '/tf/ship_data/train_ship_segmentations_OD.csv'
nb_images = 2500
boat_rate = 0.5
cut_rate = 0.5
tfrecord_dir = '/tf/ship_detect_tl/data'

generate_tf_record(path_images, path_to_csv, nb_images, cut_rate, boat_rate, tfrecord_dir)

