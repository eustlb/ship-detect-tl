import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

def configure_pipeline(pipeline_config_path, config_dict):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    
    with tf.io.gfile.GFile(pipeline_config_path, "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)    

    pipeline_config.model.ssd.num_classes = config_dict['num_classes']
    pipeline_config.train_config.batch_size = config_dict['batch_size']
    pipeline_config.train_config.fine_tune_checkpoint = config_dict['fine_tune_checkpoint']
    pipeline_config.train_config.fine_tune_checkpoint_type = config_dict['fine_tune_checkpoint_type']
    pipeline_config.train_input_reader.label_map_path= config_dict['label_map_path']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [config_dict['train_tfrecord_path']]
    pipeline_config.eval_input_reader[0].label_map_path = config_dict['label_map_path']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [config_dict['test_tfrecord_path']]

    config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
    with tf.io.gfile.GFile(pipeline_config_path, "wb") as f:                                                                                                                                                                                                                     
        f.write(config_text)   



