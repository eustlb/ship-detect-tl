import os
import shutil

paths = os.listdir('/tf/ship_data/train_v2/train')

for i in range(1000):
    print(i)
    path = paths[i]

    src = '/tf/ship_data/train_v2/train/' + path
    dest = '/tf/ship_data_lite/'+ os.path.basename(path)

    shutil.copyfile(src, dest) # on déplace l'image