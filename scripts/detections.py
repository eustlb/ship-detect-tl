import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file('/tf/pretrained_models/checkpoints/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8/pipeline.config')
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('/tf/custom_models/faster_rcnn_resnet101_v1_640x640_coco17_tpu-8/ckpt-10')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap('/tf/ship_detect_tl/data/label_map.txt')

for image_name in os.listdir('/tf/ship_data/train_v2')[-5:]:

    img = cv2.imread('/tf/ship_data/train_v2/'+image_name)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    print(detections)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

                

    im = Image.fromarray(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    im.save('/tf/predictions/'+image_name)

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