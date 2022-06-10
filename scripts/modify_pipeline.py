from pyexpat import model
from object_detection.utils import config_util
from object_detection import model_lib_v2
import os 

def modify_pipeline(pipeline_config_path, config_dict):
    """
    Permet de modifier le fichier pipeline.config d'un modèle donné à partir d'un dictionnaire dont les clefs correspondent aux paramètres de base de ce fichier .config.

    :param pipeline_config_path: str, path du fichier pipeline.config que l'on souhaite modifier.
    :param config_dict: dict, dictionnaire qui sert à cette modification. Sa strucure doit être (strictement) {'num_classes':,'batch_size':,'train_tfrecord_path':,'test_tfrecord_path':,'label_map_path':,'fine_tune_checkpoint':,'fine_tune_checkpoint_type':}.
    :return: Void.
    """

    # Convertir le fichier config original en un dictionnaire.
    pipeline_config_dict = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    # Changer la valeur des clefs concernées.
    config_util._update_batch_size(pipeline_config_dict, config_dict['batch_size'])
    config_util._update_num_classes(pipeline_config_dict["model"], config_dict['num_classes'])
    config_util._update_label_map_path(pipeline_config_dict, config_dict['label_map_path'])
    pipeline_config_dict['train_input_config'].tf_record_input_reader.input_path[0] = config_dict['train_tfrecord_path']
    pipeline_config_dict['eval_input_config'].tf_record_input_reader.input_path[0] = config_dict['test_tfrecord_path']
    pipeline_config_dict['train_config'].fine_tune_checkpoint = config_dict['fine_tune_checkpoint']
    pipeline_config_dict['train_config'].fine_tune_checkpoint_type = config_dict['fine_tune_checkpoint_type']

    # Convertir ce dictionnaire en un protobuf.
    pipeline_config = config_util.create_pipeline_proto_from_configs(pipeline_config_dict)

    # Remplacer le pipeline.config par ce nouveau pipeline.config contenant les valeurs souhaitées.
    config_util.save_pipeline_config(pipeline_config, os.path.dirname(pipeline_config_path))
