import tensorflow as tf 

def check_tfrecord(tfrecord_path):
    """
    Permet d'afficher la première valeur du tfrecord analysé. Aussi, la valeur de la clef 'image/encoded' ne sera pas affichée dans le terminal
    parce qu'il s'agit de l'image encodée, beaucoup trop longue à afficher et qui ne nous intéresse pas ici.

    :param tfrecord_path: str, path du tfrecord à analyser.
    :return: Void.
    """

    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        for key, feature in example.features.feature.items():
            if key != 'image/encoded':
                print(key)
                print(feature)

if __name__=='__main__':
    tfrecord_path = '/tf/ship_data/annotations/100_80_90/train_100_80_90.tfrecord'
    check_tfrecord(tfrecord_path)