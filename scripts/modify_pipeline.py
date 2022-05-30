from pyexpat import model
from object_detection.utils import config_util
from object_detection import model_lib_v2
import os 

def modify_pipeline(pipeline_config_path, config_dict):

    # Load the pipeline config as a dictionary
    pipeline_config_dict = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    # override config parameters 
    config_util._update_batch_size(pipeline_config_dict, config_dict['batch_size'])
    config_util._update_num_classes(pipeline_config_dict["model"], config_dict['num_classes'])
    config_util._update_label_map_path(pipeline_config_dict, config_dict['label_map_path'])

    pipeline_config_dict['train_input_config'].tf_record_input_reader.input_path[0] = config_dict['train_tfrecord_path']
    pipeline_config_dict['eval_input_config'].tf_record_input_reader.input_path[0] = config_dict['test_tfrecord_path']
    pipeline_config_dict['train_config'].fine_tune_checkpoint = config_dict['fine_tune_checkpoint']
    pipeline_config_dict['train_config'].fine_tune_checkpoint_type = config_dict['fine_tune_checkpoint_type']

    # Convert the pipeline dict back to a protobuf object
    pipeline_config = config_util.create_pipeline_proto_from_configs(pipeline_config_dict)

    # EXAMPLE USAGE:
    config_util.save_pipeline_config(pipeline_config, os.path.dirname(pipeline_config_path))
