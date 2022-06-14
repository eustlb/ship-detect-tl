import os
from charset_normalizer import detect
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file('/tf/custom_models/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('/tf/custom_models/faster_rcnn_resnet152_v1_1024x1024_coco17_tpu-8/checkpoint/ckpt-70')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap('/tf/ship_detect_tl/data/label_map.txt')
diff_images_list = ['0c939e612.jpg','0d136fde3.jpg','1dfce9923.jpg','1feb020df.jpg','fb2df7e9b.jpg','f427269bf.jpg','f221ea400.jpg','f72d6ba13.jpg','f47e91719.jpg','dd22f309e.jpg','d85469648.jpg','d482d562b.jpg','0b22c4092.jpg']
# diff_images_list = ['0adc9c314.jpg']

# for image_name in tqdm(os.listdir('/tf/ship_data/train_v2')[-1000:]):
for image_name in tqdm(diff_images_list):
# for image_name in tqdm(pd.read_csv('/tf/ship_data/annotations/100_80_90/test_100_80_90.csv')['filename'].unique()[:100]):

    image_np = cv2.imread('/tf/ship_data/train_v2/'+image_name)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    # image_np = np.array(image_np)


    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    # truth image
    image_truth = image_np.copy()
    labels_df = pd.read_csv('/tf/ship_data/train_ship_segmentations_OD.csv')
    width, height = 768, 768

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    detections_bbox = []
    detections_scores = np.array([])
    detections_classes = np.array([])

    df = labels_df.loc[labels_df['filename'] == image_name]

    for index, row in df.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)

    for i in range(len(xmaxs)):
        l = [ymaxs[i],xmins[i],ymins[i],xmaxs[i]]
        detections_bbox.append(l)
        detections_scores = np.append(detections_scores, 1)
        detections_classes = np.append(detections_classes, 0)
    
    detections_bbox = np.array(detections_bbox)


    # detection_classes should be ints.
    detections_classes = detections_classes.astype(np.int64)
    
    # draw predicted boxes on image
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=.5,
                agnostic_mode=False)

    #draw predicted boxes on second image
    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_truth,
                detections_bbox,
                detections_classes+label_id_offset,
                detections_scores,
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=20,
                min_score_thresh=.5,
                agnostic_mode=False)

    # image_predicted = cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB)
    # image_truth = cv2.cvtColor(image_truth, cv2.COLOR_BGR2RGB)

    image_predicted = image_np_with_detections
    
    vis = np.concatenate((image_predicted, image_truth), axis=1)
       
    im = Image.fromarray(vis)
    im.save('/tf/predictions/'+image_name[:image_name.index('.')]+'.png')

# plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
# plt.savefig('/tf/archive/test.png')

# def select_bboxs(detections, threshold):
#     """
#     Return the list of selected bboxs according to threshold. dectections is the output tensor of the prediction.
#     """
#     scores = detections['detection_scores'].numpy()[0]
#     bbox_selected = []

#     for i in range(len(scores)):
#         if scores[i]>threshold:
#             bbox = detections['detection_boxes'].numpy()[0][i]
#             bbox_selected.append(bbox)

#     return bbox_selected

# bboxs = select_bboxs(detections,0.3)

# def save_image_bbox(path, bboxs):
#     image = Image.open('/tf/archive/test/raccoon-28.jpg')
#     h, w = image.size
#     draw = ImageDraw.Draw(image)
#     for bbox in bboxs :
#         x0 = bbox[1]*w
#         y0 = bbox[0]*h
#         x1 = bbox[3]*w
#         y1 = bbox[2]*h
#         print(x0,y0,x1,y1)
#         draw.rectangle([x0,y0,x1,y1], outline='#20d200')
#     image.save(path)

# save_image_bbox('/tf/archive/test.png', bboxs)