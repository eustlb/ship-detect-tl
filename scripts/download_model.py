import os
from tensorflow.python.keras.utils.data_utils import get_file
import shutil


def download_model(model_url, models_dir):

    working_dir = os.path.dirname(models_dir)

    file_name = os.path.basename(model_url)
    model_name = file_name[:file_name.index('.')]
    
    if os.path.exists(os.path.join(models_dir,'checkpoints',model_name)):
        print('Le modèle a déjà été téléchargé auparavant.')
        return
    
    os.makedirs(models_dir, exist_ok=True) 
    get_file(fname=file_name, origin=model_url, cache_subdir='checkpoints', cache_dir=models_dir, extract=True)

    os.makedirs(os.path.join(working_dir,'custom_models',model_name), exist_ok=True)
    os.makedirs(os.path.join(working_dir,'custom_models',model_name,'checkpoint'), exist_ok=True)
    shutil.copyfile(os.path.join(models_dir,'checkpoints',model_name,'pipeline.config'), os.path.join(working_dir,'custom_models',model_name,'pipeline.config'))

