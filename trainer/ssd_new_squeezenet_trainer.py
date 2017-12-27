# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/18

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

from keras import backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

from ssd.SSDLoss import SSDLoss
from ssd.SSDBoxEncoder import SSDBoxEncoder
from ssd.BatchGenerator import BatchGenerator
from ssd.Layer_AnchorBoxes import AnchorBoxes
from ssd.Layer_L2Normalization import L2Normalization
from ssd.box_encode_decode_utils import decode_y, decode_y2
from ssd.feature_base_new_squeezenet import base_feature_model as squeezenet_300


img_height = 300 # Height of the input images
img_width = 300 # Width of the input images
img_channels = 3 # Number of color channels of the input images
n_classes = 21 # Number of classes including the background class, e.g. 21 for the Pascal VOC datasets
scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets, the factors for the MS COCO dataset are smaller, namely [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
aspect_ratios = [[0.5, 1.0, 2.0],
                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                 [1.0/3.0, 0.5, 1.0, 2.0, 3.0],
                 [0.5, 1.0, 2.0],
                 [0.5, 1.0, 2.0]] # The anchor box aspect ratios used in the original SSD300
two_boxes_for_ar1 = True
limit_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation
coords = 'centroids' # Whether the box coordinates to be used as targets for the model should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = True

# 1: Build the Keras model
K.clear_session() # Clear previous models from memory.
model, predictor_sizes = squeezenet_300(image_size=(img_height, img_width, img_channels),
                                 n_classes=n_classes,
                                 min_scale=None, # You could pass a min scale and max scale instead of the `scales` list, but we're not doing that here
                                 max_scale=None,
                                 scales=scales,
                                 aspect_ratios_global=None,
                                 aspect_ratios_per_layer=aspect_ratios,
                                 two_boxes_for_ar1=two_boxes_for_ar1,
                                 limit_boxes=limit_boxes,
                                 variances=variances,
                                 coords=coords,
                                 normalize_coords=normalize_coords)

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

model.summary()
sys.exit(0)

# 2: Load the trained model weights into the model.
# TODO: Set the path to the model weights.
# K.clear_session() # Clear previous models from memory.
# model_path = ''
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'compute_loss': ssd_loss.compute_loss})

# TODO: Set the path to the `.h5` file of the model to be loaded.
# model_path = 'ssd300_0.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
# ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
# K.clear_session() # Clear previous models from memory.
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'compute_loss': ssd_loss.compute_loss})

# 3: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.
# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
ssd_box_encoder = SSDBoxEncoder(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=None,
                                max_scale=None,
                                scales=scales,
                                aspect_ratios_global=None,
                                aspect_ratios_per_layer=aspect_ratios,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)

### Set up the data generators for the training
# 1: Instantiate to `BatchGenerator` objects: One for training, one for validation.
train_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])
val_dataset = BatchGenerator(box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'])

# 2: Parse the image and label lists for the training and validation datasets. This can take a while.
# The directories that contain the images.
dataset_prefix = "/Volumes/projects/repos/VOC.SOURCE.DATA/"
VOC_2007_images_path      = dataset_prefix + 'VOCdevkit/VOC2007/JPEGImages/'
VOC_2007_test_images_path = dataset_prefix + 'VOCdevkit/VOC2007/JPEGImages/'
VOC_2012_images_path      = dataset_prefix + 'VOCdevkit/VOC2012/JPEGImages/'

# The directories that contain the annotations.
VOC_2007_annotations_path      = dataset_prefix + 'VOCdevkit/VOC2007/Annotations/'
VOC_2007_test_annotations_path = dataset_prefix + 'VOCdevkit/VOC2007/Annotations/'
VOC_2012_annotations_path      = dataset_prefix + 'VOCdevkit/VOC2012/Annotations/'

# The paths to the image sets.
VOC_2007_train_image_set_path    = dataset_prefix + 'VOCdevkit/VOC2007/ImageSets/Main/train.txt'
VOC_2012_train_image_set_path    = dataset_prefix + 'VOCdevkit/VOC2012/ImageSets/Main/train.txt'
VOC_2007_val_image_set_path      = dataset_prefix + 'VOCdevkit/VOC2007/ImageSets/Main/val.txt'
VOC_2012_val_image_set_path      = dataset_prefix + 'VOCdevkit/VOC2012/ImageSets/Main/val.txt'
VOC_2007_trainval_image_set_path = dataset_prefix + 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
VOC_2012_trainval_image_set_path = dataset_prefix + 'VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
VOC_2007_test_image_set_path     = dataset_prefix + 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

train_dataset.parse_xml(images_paths=[VOC_2007_images_path,
                                      VOC_2007_test_images_path,
                                      VOC_2012_images_path],
                        annotations_paths=[VOC_2007_annotations_path,
                                           VOC_2007_test_annotations_path,
                                           VOC_2012_annotations_path],
                        image_set_paths=[VOC_2007_trainval_image_set_path,
                                         VOC_2007_test_image_set_path,
                                         VOC_2012_train_image_set_path],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False)

val_dataset.parse_xml(images_paths=[VOC_2012_images_path],
                      annotations_paths=[VOC_2012_annotations_path],
                      image_set_paths=[VOC_2012_val_image_set_path],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=False,
                      ret=False)

# 4: Set the batch size.
batch_size = 64 # Change the batch size if you like, or if you run into memory issues with your GPU.

# 5: Set the image processing / data augmentation options and create generator handles.
train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         train=True,
                                         ssd_box_encoder=ssd_box_encoder,
                                         equalize=False,
                                         brightness=(0.5, 2, 0.5),
                                         flip=0.5,
                                         translate=False,
                                         scale=False,
                                         max_crop_and_resize=(img_height, img_width, 1, 3), # This one is important because the Pascal VOC images vary in size
                                         full_crop_and_resize=(img_height, img_width, 1, 3, 0.5), # This one is important because the Pascal VOC images vary in size
                                         random_crop=False,
                                         crop=False,
                                         resize=False,
                                         gray=False,
                                         limit_boxes=True, # While the anchor boxes are not being clipped, the ground truth boxes should be
                                         include_thresh=0.4,
                                         diagnostics=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=True,
                                     train=True,
                                     ssd_box_encoder=ssd_box_encoder,
                                     equalize=False,
                                     brightness=False,
                                     flip=False,
                                     translate=False,
                                     scale=False,
                                     max_crop_and_resize=(img_height, img_width, 1, 3), # This one is important because the Pascal VOC images vary in size
                                     full_crop_and_resize=(img_height, img_width, 1, 3, 0.5), # This one is important because the Pascal VOC images vary in size
                                     random_crop=False,
                                     crop=False,
                                     resize=False,
                                     gray=False,
                                     limit_boxes=True,
                                     include_thresh=0.4,
                                     diagnostics=False)

# Get the number of samples in the training and validations datasets to compute the epoch lengths below.
n_train_samples = train_dataset.get_n_samples()
n_val_samples   = val_dataset.get_n_samples()


# Define a learning rate schedule.
def lr_schedule(epoch):
    if epoch <= 300:
        return 0.001
    elif epoch <= 800:
        return 0.0001
    else:
        return 0.00001

# TODO: Set the number of epochs to train for.
epochs = 1000

history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = ceil(n_train_samples/batch_size),
                              epochs = epochs,
                              callbacks = [ModelCheckpoint('weights/squeezenet300_model_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
                                                           monitor='val_loss',
                                                           verbose=1,
                                                           save_best_only=True,
                                                           save_weights_only=False,
                                                           mode='auto',
                                                           period=1),
                                           LearningRateScheduler(lr_schedule),
                                           EarlyStopping(monitor='val_loss',
                                                         min_delta=0.00001,
                                                         patience=800)],
                              validation_data = val_generator,
                              validation_steps = ceil(n_val_samples/batch_size))

# TODO: Set the filename (without the .h5 file extension!) under which to save the model and weights.
#       Do the same in the `ModelCheckpoint` callback above.
model_name = 'squeezenet300'
model.save('weights/{}.h5'.format(model_name))
model.save_weights('weights/{}_weights.h5'.format(model_name))

print()
print("Model saved under {}.h5".format(model_name))
print("Weights also saved separately under {}_weights.h5".format(model_name))
print()


### Make predictions
def make_predictions():
    # 1: Set the generator
    predict_generator = val_dataset.generate(batch_size=1,
                                             shuffle=True,
                                             train=False,
                                             equalize=False,
                                             brightness=False,
                                             flip=False,
                                             translate=False,
                                             scale=False,
                                             max_crop_and_resize=(300, 300, 1, 3),
                                             full_crop_and_resize=(300, 300, 1, 3, 0.5),
                                             random_crop=False,
                                             crop=False,
                                             resize=False,
                                             gray=False,
                                             limit_boxes=True,
                                             include_thresh=0.4,
                                             diagnostics=False)

    # 2: Generate samples
    X, y_true, filenames = next(predict_generator)
    i = 0 # Which batch item to look at

    print("Image:", filenames[i])
    print()
    print("Ground truth boxes:\n")
    print(y_true[i])

    # 3: Make a prediction
    y_pred = model.predict(X)

    # 4: Decode the raw prediction `y_pred`
    y_pred_decoded = decode_y2(y_pred,
                               confidence_thresh=0.5,
                               iou_threshold=0.4,
                               top_k='all',
                               input_coords='centroids',
                               normalize_coords=normalize_coords,
                               img_height=img_height,
                               img_width=img_width)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print(y_pred_decoded[i])

    # 5: Draw the predicted boxes onto the image

    plt.figure(figsize=(20,12))
    plt.imshow(X[i])

    current_axis = plt.gca()

    for box in y_pred_decoded[i]:
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        current_axis.add_patch(plt.Rectangle((box[2], box[4]), box[3]-box[2], box[5]-box[4], color='blue', fill=False, linewidth=2))  
        current_axis.text(box[2], box[4], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})

    for box in y_true[i]:
        label = '{}'.format(classes[int(box[0])])
        current_axis.add_patch(plt.Rectangle((box[1], box[3]), box[2]-box[1], box[4]-box[3], color='green', fill=False, linewidth=2))  
        current_axis.text(box[1], box[3], label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
