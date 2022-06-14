import os
import cv2 
import tensorflow as tf
import numpy as np
import pandas as pd
from object_detection.builders import model_builder
from object_detection.utils import config_util
from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageDraw
from tqdm import tqdm

def save_image_bbox(img_path, saving_path, bboxs, scores):
    """
    Dessine sur une image des bboxs avec leurs scores et place à droite de cette image l'image avec les bboxs véritables (annotations de départ).
    :param img_path: str, path de l'image à éditer
    :param saving_path: str, path où sera enregistrée l'image 
    :param bboxs: list, liste de bboxs au format [[ymin, xmin, ymax, xmax]]
    :param scores: list, liste des scores au format [0.8]
    :return: Void.
    """
    # Placing the bboxs and scores on the image
    image = Image.open(img_path)
    h, w = image.size
    draw = ImageDraw.Draw(image)
    for i in range(len(bboxs)) :
        x0 = bboxs[i][1]*w
        y0 = bboxs[i][0]*h
        x1 = bboxs[i][3]*w
        y1 = bboxs[i][2]*h
        font_file = "/usr/local/share/fonts/bebas_neue/BebasNeue-Regular.ttf"
        font_size = 12
        font = ImageFont.truetype(font_file, font_size)
        text = str(round(scores[i]*100))+'%'
        height = font.getsize(text)[1]
        draw.rectangle([x0,y0,x1,y1], outline='#20d200')
        draw.text((x0,y0-height-3), text, font = font, fill=(32, 210, 0))

    # Do the same with truth image
    image_truth = Image.open(img_path)
    h, w = image.size
    draw_truth = ImageDraw.Draw(image_truth)

    labels_df = pd.read_csv('/tf/ship_data/train_ship_segmentations_OD.csv')

    detections_bboxs = []
    detections_scores = []

    image_name = os.path.basename(img_path)
    df = labels_df.loc[labels_df['filename'] == image_name]

    for index, row in df.iterrows():
        l = [row['ymin']/h, row['xmin']/w, row['ymax']/h, row['xmax']/w]
        detections_bboxs.append(l)
        detections_scores.append(1.)

    for i in range(len(detections_bboxs)) :
        x0 = detections_bboxs[i][1]*w
        y0 = detections_bboxs[i][0]*h
        x1 = detections_bboxs[i][3]*w
        y1 = detections_bboxs[i][2]*h
        font_file = "/usr/local/share/fonts/bebas_neue/BebasNeue-Regular.ttf"
        font_size = 12
        font = ImageFont.truetype(font_file, font_size)
        text = str(round(detections_scores[i]*100))+'%'
        height = font.getsize(text)[1]
        draw_truth.rectangle([x0,y0,x1,y1], outline='#20d200')
        draw_truth.text((x0,y0-height-3), text, font = font, fill=(32, 210, 0))

    # concatenate the two images
    dst = Image.new('RGB', (image.width + image_truth.width, image.height))
    dst.paste(image, (0, 0))
    dst.paste(image_truth, (image.width, 0))

    # save the result
    dst.save(saving_path, "PNG")

def predict(checkpoint_path, pipeline_path, img_names, thresh, saving_dir):
    """
    Effectue la détection sur une liste d'image à partir d'un modèle donnée et selon un seuil de confiance définit.

    :param checkpoint_path: str, path du checkpoint du modèle choisi pour effectuer les prédictions.
    :param pipeline_path: str, path du fichier pipeline.config qui définit le pipeline du modèle.
    :param img_names: list, liste des noms d'images sur lesquelles il faut effectuer la précition.
    :param thresh: float, seuil de confiance utilisé, ex: 0.8 pour 80%.
    :param saving_dir: str, répertoire où seront enregistrées les images avec prédiction.
    :return: Void.
    """
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_path)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(checkpoint_path)).expect_partial()

    for img_name in tqdm(img_names):
        # Load the image and convert it to a tensor
        img_path = '/tf/ship_data/train_v2/'+img_name
        image_np = cv2.imread(img_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

        # Run model prediction on this tensor
        image, shapes = detection_model.preprocess(input_tensor)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # Store the bboxs and scores according to the threshold in lists
        i = 0
        bboxs = []
        scores = []

        while detections['detection_scores'][i] >= thresh:
            scores.append(detections['detection_scores'][i])
            bboxs.append(detections['detection_boxes'][i])
            i+=1
    
        # save the new image with the predictions using these lists
        new_img_name = img_name[:img_name.index('.')]+'.png' # png format for higher quality
        save_image_bbox(img_path, os.path.join(saving_dir, new_img_name),bboxs,scores)
               
if __name__ == '__main__':
    checkpoint_path = '/tf/custom_models/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8/checkpoint/ckpt-92'
    pipeline_path = '/tf/custom_models/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8/pipeline.config'
    img_names = ['0c939e612.jpg','0d136fde3.jpg','1dfce9923.jpg','1feb020df.jpg','fb2df7e9b.jpg','f427269bf.jpg','f221ea400.jpg','f72d6ba13.jpg','f47e91719.jpg','dd22f309e.jpg','d85469648.jpg','d482d562b.jpg','0b22c4092.jpg']
    # img_names = pd.read_csv('/tf/ship_data/annotations/100_80_90/test_100_80_90.csv')['filename'].unique()[:100]
    thresh = 0.5
    saving_dir = '/tf/predictions'
    predict(checkpoint_path, pipeline_path, img_names, thresh, saving_dir)



