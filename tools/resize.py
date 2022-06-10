from chitra.image import Chitra
import os
import pandas as pd
import cv2


def resize(image_dir, path_original_csv, resize_format, new_image_dir):
    """
    Redimensionne des images et crée le csv au format pascal VOC correspondant.

    :param image_dir: str, répertoire où sont stockées les images à redimensionner.
    :param  path_original_csv: str, path du csv au format pasval VOC associé aux images présentes dans image_dir
    :param resize_format: tuple, len(resize_format=2). Couple de tailles en pixels dans lequels on veut redimensionner l'image.
    :param new_image_dir: str, répertoire où les nouvelles images et nouveau csv seront créés. 
    :return: Void
    """

    df = pd.read_csv(path_original_csv)

    od_dict = {'filename':[], 'width': [], 'height': [], 'class': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}

    for img_name in os.listdir(image_dir):
        a = df.loc[df['filename'] == img_name]
        for index, row in a.iterrows():
            box = [[row['xmin'],row['ymin'],row['xmax'],row['ymax']]]
            image = Chitra(os.path.join(image_dir, img_name), box, [row['class']])
            image.resize_image_with_bbox(resize_format)
            od_dict['filename'].append(img_name)
            od_dict['width'].append(768)
            od_dict['height'].append(768)
            od_dict['xmin'].append(image.bboxes[0].x1)
            od_dict['ymin'].append(image.bboxes[0].y1)
            od_dict['xmax'].append(image.bboxes[0].x2)
            od_dict['ymax'].append(image.bboxes[0].y2)
            od_dict['class'].append(image.bboxes[0].label)

        image = cv2.imread(os.path.join(image_dir, img_name))
        output = cv2.resize(image, resize_format)
        cv2.imwrite(os.path.join(new_image_dir, img_name),output)

    df_od_train= pd.DataFrame(od_dict)
    df_od_train.to_csv(os.path.join(new_image_dir, 'resized_labels_.csv', index=False))
