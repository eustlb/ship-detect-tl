import os
from tensorflow.python.keras.utils.data_utils import get_file
import shutil

def download_model(model_url, models_dir):
    """
    Télécharge le modèle choisi sur le tensorflow model zoo, le place dans le répertoire models_dir choisi et crée un répertoire 'custom_models/'
    contenant lui-même le fichier config du modèle et un sous_répertoire 'checkpoints/'. C'est ce répertoire qui sera en jeu lors de l'entrainement 
    et pour l'enregistrement des checkpoints. Le répertoire models_dir choisi est simplement un lieu de 'stockage' du modèle générique téléchargé.

    :param model_url: str, lien vers le modèle choisi du tensorflow model zoo.
    :param models_dir: str, répertoire où l'on souhaite que le modèle soit téléchargé. 
    :return: Void.
    """

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

