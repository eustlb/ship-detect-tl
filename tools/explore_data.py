import pandas as pd

data_label = pd.read_csv('/tf/ship_data/train_ship_segmentations_v2.csv')

print(len(data_label))

images_names = data_label['ImageId'].unique()
nb_images_total = len(images_names)

nb_images_bateau = nb_images_total - len(data_label[data_label.EncodedPixels != data_label.EncodedPixels])

print(
    "Nombre d'images : {}, dont {} comprenant au moins un bateau.".format(nb_images_total, nb_images_bateau)
)