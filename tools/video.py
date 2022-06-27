import os
import cv2
from cv2 import resize
from sklearn.ensemble import AdaBoostRegressor 
import tensorflow as tf
import numpy as np
import pandas as pd
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
from PIL import Image, ImageDraw, ImageFont
from PIL import Image, ImageDraw
from tqdm import tqdm


def save_image_bbox(img_path, saving_path, bboxs, scores):
    """
    Dessine sur une image des bboxs avec leurs scores.

    :param img_path: str, path de l'image à éditer
    :param saving_path: str, path où sera enregistrée l'image 
    :param bboxs: list, liste de bboxs au format [[ymin, xmin, ymax, xmax]]
    :param scores: list, liste des scores au format [0.8]
    :return: Void.
    """
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
    image.save(saving_path, "PNG")

def predict(checkpoint_path, pipeline_path, img_dir, img_names, thresh, saving_dir):
    """
    Effectue la détection sur une liste d'image à partir d'un modèle donnée et selon un seuil de confiance définit.

    :param checkpoint_path: str, path du checkpoint du modèle choisi pour effectuer les prédictions.
    :param pipeline_path: str, path du fichier pipeline.config qui définit le pipeline du modèle.
    :param img_dir: str, répertoire où sont stockées les images
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
        img_path = os.path.join(img_dir, img_name)
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
        if len(bboxs)!=0:
            save_image_bbox(img_path, os.path.join(saving_dir, new_img_name),bboxs,scores)

def predict2(checkpoint_path, pipeline_path, img_dir, img_names, thresh, saving_dir):
    """
    Effectue la détection sur une liste d'image à partir d'un modèle donnée et selon un seuil de confiance définit.

    :param checkpoint_path: str, path du checkpoint du modèle choisi pour effectuer les prédictions.
    :param pipeline_path: str, path du fichier pipeline.config qui définit le pipeline du modèle.
    :param img_dir: str, répertoire où sont stockées les images
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

    for image_name in tqdm(img_names):

        # Load the image and convert it to a tensor
        img_path = os.path.join(img_dir, image_name)
        image_np = cv2.imread(img_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        image, shapes = detection_model.preprocess(input_tensor)
        prediction_dict = detection_model.predict(input_tensor, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        category_index = label_map_util.create_category_index_from_labelmap('/tf/ship_detect_tl/data/label_map.txt')
        
        # draw predicted boxes on image
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=20,
                    min_score_thresh=thresh,
                    agnostic_mode=False)

        im = Image.fromarray(image_np_with_detections)
        im.save(os.path.join(saving_dir, image_name[:image_name.index('.')]+'.png'))

def extract_im_video(video_path, saving_dir, nb_images):
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,250000)      # just cue to 20 sec. position
    success,image = vidcap.read()
    count = 0
    while count<nb_images:
        cv2.imwrite(os.path.join(saving_dir, 'frame{}.jpg'.format(count)), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1

def adapt_frames(origin_frames_dir, saving_dir):
    print('Modification des images')
    for im in tqdm(os.listdir(origin_frames_dir)):
        image = Image.open(os.path.join(origin_frames_dir,im))
        image = image.crop((420,0,1500,1080))
        # image = image.resize((768,768))
        image.save(os.path.join(saving_dir,'crop_'+im))

def create_video(video_dir, saving_dir):
    from moviepy.editor import ImageSequenceClip
    files = [os.path.join(video_dir,vid_name) for vid_name in os.listdir(video_dir)]
    clip = ImageSequenceClip(files, fps = 30) 
    clip.write_videofile(os.path.join(saving_dir,"video.mp4"))

if __name__ == '__main__':
    video_path = '/tf/ship_data/video/images_drone_brise_lame/vol_drone.MP4'
    saving_dir = '/tf/video'
    nb_images = 900
    # extract_im_video(video_path,saving_dir,nb_images)

    origin_frames_dir = '/tf/video'
    saving_dir = '/tf/video_crop'
    # adapt_frames(origin_frames_dir, saving_dir)

    checkpoint_path = '/tf/custom_models/faster_rcnn_resnet101_6/checkpoint/ckpt-11'
    pipeline_path = '/tf/custom_models/faster_rcnn_resnet101_6/pipeline.config'
    img_dir = '/tf/video_crop'
    img_names = os.listdir(img_dir)
    thresh = 0.3
    saving_dir = '/tf/video_predi'
    predict(checkpoint_path, pipeline_path, img_dir, img_names, thresh, saving_dir)

    video_dir = '/tf/video_predi'
    saving_dir = '/tf/tests'
    create_video(video_dir,saving_dir)

    








