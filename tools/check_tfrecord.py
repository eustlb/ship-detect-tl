import tensorflow as tf 
raw_dataset = tf.data.TFRecordDataset('/tf/annotations/test_13250_90_70.tfrecord')

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    for key, feature in example.features.feature.items():
        if key != 'image/encoded':
            print(key)
            print(feature)