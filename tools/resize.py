from chitra.image import Chitra
import os
import pandas as pd
import cv2

df = pd.read_csv('/tf/archive/train_labels_.csv')

od_dict = {'filename':[], 'width': [], 'height': [], 'class': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}

for img_name in os.listdir('/tf/archive/Racoon/images'):
    a = df.loc[df['filename'] == img_name]
    for index, row in a.iterrows():
        box = [[row['xmin'],row['ymin'],row['xmax'],row['ymax']]]
        image = Chitra(os.path.join('/tf/archive/Racoon/images', img_name), box, [row['class']])
        image.resize_image_with_bbox((768, 768))
        od_dict['filename'].append(img_name)
        od_dict['width'].append(768)
        od_dict['height'].append(768)
        od_dict['xmin'].append(image.bboxes[0].x1)
        od_dict['ymin'].append(image.bboxes[0].y1)
        od_dict['xmax'].append(image.bboxes[0].x2)
        od_dict['ymax'].append(image.bboxes[0].y2)
        od_dict['class'].append(image.bboxes[0].label)

    image = cv2.imread(os.path.join('/tf/archive/Racoon/images', img_name))
    output = cv2.resize(image, (768, 768))
    cv2.imwrite(os.path.join('/tf/archive/resized_racoons', img_name),output)

df_od_train= pd.DataFrame(od_dict)
df_od_train.to_csv('/tf/archive/resized_train_labels_.csv', index=False)
