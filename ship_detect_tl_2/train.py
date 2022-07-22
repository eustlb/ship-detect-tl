from configparser import Interpolation
import segmentation_models as sm
sm.set_framework('tf.keras')

# load data

# preprocess input

# define model 
model = sm.FPN(backbone_name='restnet34', input_shape = (None, None, 3), encoder_weights ='imagenet', freeze_encoder = False, fpn_layers='default',
               pyramid_block_filters=256, segmentation_block_filters=128, upsample_rates=(2,2,2), last_upsample=4, Interpolation='bilinear', 
               use_batchnorm=True, classes=1, activation='sigmoid')

print(type(model))
# model.compile('Adam', loss=sm.bce_jaccard_loss, metrics=[sm.iou_score])

# fit model
# model.

