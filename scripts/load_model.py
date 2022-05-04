import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
import os
from tensorflow.python.keras.utils.data_utils import get_file

class Model:
    def __init__(self):
        pass

    def download_model(self, model_url):
        file_name = os.path.basename(model_url)
        self.model_name = file_name[:file_name.index('.')]
        self.cache_dir = './pretrained_models'
        os.makedirs(self.cache_dir, exist_ok=True)
        get_file(fname=file_name, origin=model_url, cache_dir=self.cache_dir, cache_subdir='checkpoints', extract=True)

    def load_model(self):
        print('Loading Model ' + self.model_name)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cache_dir, 'checkpoints', self.model_name, 'saved_model'))
        print('Model ' + self.model_name + ' loaded successfully...')