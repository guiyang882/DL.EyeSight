# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/19

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
该文件主要是描述SSD中基础特征提取模型
"""
import numpy as np

from keras.models import Model
from keras.layers import Input, Activation, Conv2D, Lambda, Reshape, Concatenate
from keras.layers import MaxPooling2D, BatchNormalization

from eagle.brain.ssd.Layer_AnchorBoxes import AnchorBoxes


def _fire(x, filters, name="fire"):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Conv2D(sq_filters, (1, 1), activation="relu", padding="same", kernel_initializer="he_normal", name=name+"/squeeze1x1")(x)
    expand1 = Conv2D(ex1_filters, (1, 1), activation="relu", padding="same", kernel_initializer="he_normal", name=name+"/expand1x1")(squeeze)
    expand2 = Conv2D(ex2_filters, (3, 3), activation="relu", padding="same", kernel_initializer="he_normal", name=name+"/expand3x3")(squeeze)
    x = Concatenate(axis=-1, name=name+"/concate")([expand1, expand2])
    return x

def _fire_with_bn(x, filters, name="fire"):
    sq_filters, ex1_filters, ex2_filters = filters
    squeeze = Conv2D(sq_filters, (1, 1), activation="relu", padding="same", kernel_initializer="he_normal", name=name+"/squeeze1x1")(x)
    expand1 = Activation(activation="relu", name=name+"/relu_expand1x1")(BatchNormalization(name=name+"/expand1x1/bn")(Conv2D(ex1_filters, (1, 1), strides=(1, 1), padding="same", kernel_initializer="he_normal", name=name+"/expand1x1")(squeeze)))
    expand2 = Activation(activation="relu", name=name+"/relu_expand3x3")(BatchNormalization(name=name+"/expand3x3/bn")(Conv2D(ex2_filters, (3, 3), strides=(1, 1), padding="same", kernel_initializer="he_normal", name=name+"/expand3x3")(squeeze)))
    x = Concatenate(axis=-1, name=name+"/concate")([expand1, expand2])
    return x

def _conv2D_with_bn(x, n_filters, k_size, k_stride, name, pad="same"):
    x = Conv2D(n_filters, k_size, strides=(k_stride, k_stride), padding=pad, kernel_initializer="he_normal", name=name+"/conv")(x)
    x = BatchNormalization(name=name+"/bn")(x)
    x = Activation(activation="relu", name=name+"/relu")(x)
    return x

def base_feature_model(image_size, n_classes,
                       min_scale=0.1, max_scale=0.9, scales=None,
                       aspect_ratios_global=None, aspect_ratios_per_layer=None,
                       two_boxes_for_ar1=True, limit_boxes=False,
                       variances=[0.1, 0.1, 0.2, 0.2],
                       coords='centroids',
                       normalize_coords=False):
    n_predictor_layers = 6  # The number of predictor conv layers in the network is 6 for the original SSD300
    # Get a few exceptions out of the way first
    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    pl_aspect_ratios = [0] * n_predictor_layers
    if aspect_ratios_per_layer:
        for idx in range(0, n_predictor_layers):
            pl_aspect_ratios[idx] = aspect_ratios_per_layer[idx]
    else:
        for idx in range(0, n_predictor_layers):
            pl_aspect_ratios[idx] = aspect_ratios_global

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    pl_n_boxes = [0] * n_predictor_layers
    if aspect_ratios_per_layer:
        n_boxes = []
        for aspect_ratios in aspect_ratios_per_layer:
            if (1 in aspect_ratios) & two_boxes_for_ar1:
                n_boxes.append(len(aspect_ratios) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(aspect_ratios))
        for idx in range(0, n_predictor_layers):
            pl_n_boxes[idx] = n_boxes[idx]
        # n_boxes_conv4_3 = n_boxes[0] # 4 boxes per cell for the original implementation
        # n_boxes_fc7 = n_boxes[1] # 6 boxes per cell for the original implementation
        # n_boxes_conv6_2 = n_boxes[2] # 6 boxes per cell for the original implementation
        # n_boxes_conv7_2 = n_boxes[3] # 6 boxes per cell for the original implementation
        # n_boxes_conv8_2 = n_boxes[4] # 4 boxes per cell for the original implementation
        # n_boxes_conv9_2 = n_boxes[5] # 4 boxes per cell for the original implementation
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        for idx in range(0, n_predictor_layers):
            pl_n_boxes[idx] = n_boxes
        # n_boxes_conv4_3 = n_boxes
        # n_boxes_fc7 = n_boxes
        # n_boxes_conv6_2 = n_boxes
        # n_boxes_conv7_2 = n_boxes
        # n_boxes_conv8_2 = n_boxes
        # n_boxes_conv9_2 = n_boxes

    # Input image format
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ### Design the actual network
    x = Input(shape=(img_height, img_width, img_channels))
    normed = Lambda(
        lambda z: z/127.5 - 1.0,
        output_shape=(img_height, img_width, img_channels),
        name='lambda1')(x)

#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#

    conv1 = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", kernel_initializer="he_normal", name="conv1")(normed)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool1", padding="same")(conv1)

    dilated_conv11 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding="same", activation="relu", kernel_initializer="he_normal", name="dilated_conv11")(conv1)
    dilated_pool11 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="dilated_pool11", padding="same")(dilated_conv11)

    dilated_conv12 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding="same", activation="relu", kernel_initializer="he_normal", name="dilated_conv12")(conv1)
    dilated_pool12 = MaxPooling2D(pool_size=(3, 3), strides=(4, 4), name="dilated_pool12", padding="same")(dilated_conv12)

    dilated_conv13 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(8, 8), padding="same", activation="relu", kernel_initializer="he_normal", name="dilated_conv13")(conv1)
    dilated_pool13 = MaxPooling2D(pool_size=(3, 3), strides=(8, 8), name="dilated_pool13", padding="same")(dilated_conv13)

#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#
    
    fire2 = _fire(pool1, (16, 32, 32), name="fire2")
    fire3 = _fire(fire2, (16, 32, 32), name="fire3")
    concat_d11_f3 = Concatenate(name='concat_d11_f3')([dilated_pool11, fire3])
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool3", padding="same")(concat_d11_f3)

    dilated_conv21 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding="same", activation="relu", kernel_initializer="he_normal", name="dilated_conv21")(concat_d11_f3)
    dilated_pool21 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="dilated_pool21", padding="same")(dilated_conv21)

    dilated_conv22 = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(4, 4), padding="same", activation="relu", kernel_initializer="he_normal", name="dilated_conv22")(concat_d11_f3)
    dilated_pool22 = MaxPooling2D(pool_size=(3, 3), strides=(4, 4), name="dilated_pool22", padding="same")(dilated_conv22)

#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#

    fire4 = _fire(pool3, (32, 64, 64), name="fire4")
    fire5 = _fire(fire4, (32, 64, 64), name="fire5")
    concat_d12_f5 = Concatenate(name='concat_d12_f5')([dilated_pool12, fire5])
    concat_d21_d12_f5 = Concatenate(name='concat_d21_d12_f5')([dilated_pool21, concat_d12_f5])
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5', padding="same")(concat_d21_d12_f5)

    fire5_conv_bn = Conv2D(256, (3, 3), strides=(1, 1), kernel_initializer="he_normal", name="fire5_conv_bn")(concat_d21_d12_f5)
    fire5_bn = BatchNormalization(name="fire5_bn")(fire5_conv_bn)

    dilated_conv31 = Conv2D(96, kernel_size=(3, 3), strides=(1, 1), dilation_rate=(2, 2), padding="same", activation="relu", kernel_initializer="he_normal", name="dilated_conv31")(concat_d21_d12_f5)
    dilated_pool31 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="dilated_pool31", padding="same")(dilated_conv31)

#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#

    fire6 = _fire(pool5, (48, 96, 96), name="fire6")
    fire7 = _fire(fire6, (48, 96, 96), name="fire7")
    concat_d13_f7 = Concatenate(name='concat_d13_f7')([dilated_pool13, fire7])
    concat_d22_d13_f7 = Concatenate(name='concat_d22_d13_f7')([dilated_pool22, concat_d13_f7])
    concat_d31_d22_d13_f7 = Concatenate(name='concat_d31_d22_d13_f7')([dilated_pool31, concat_d22_d13_f7])

    fire8 = _fire(concat_d31_d22_d13_f7, (96, 256, 256), name="fire8")
    fire9 = _fire_with_bn(fire8, (96, 256, 256), name="fire9")
    pool9 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool9', padding="same")(fire9)

#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#

    fire10 = _fire_with_bn(pool9, (96, 256, 256), name="fire10")
    pool10 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool10', padding="same")(fire10)

#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#

    fire11 = _fire_with_bn(pool10, (96, 256, 256), name="fire11")
    conv12_1 = _conv2D_with_bn(fire11, 128, (1, 1), 1, name="conv12_1")
    conv12_2 = _conv2D_with_bn(conv12_1, 256, (3, 3), 2, name="conv12_2")
    conv13_1 = _conv2D_with_bn(conv12_2, 64, (1, 1), 1, name="conv13_1")
    conv13_2 = _conv2D_with_bn(conv13_1, 128, (3, 3), 2, pad="valid", name="conv13_2")

#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#-------------#

    ### Build the convolutional predictor layers on top of the base network
    # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
    fire5_bn_mbox_conf = Conv2D(pl_n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', name='fire5_bn_mbox_conf')(fire5_bn)
    fire9_mbox_conf = Conv2D(pl_n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', name='fire9_mbox_conf')(fire9)
    fire10_mbox_conf = Conv2D(pl_n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', name='fire10_mbox_conf')(fire10)
    fire11_mbox_conf = Conv2D(pl_n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', name='fire11_mbox_conf')(fire11)
    conv12_2_mbox_conf = Conv2D(pl_n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', name='conv12_2_mbox_conf')(conv12_2)
    conv13_2_mbox_conf = Conv2D(pl_n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', name='conv13_2_mbox_conf')(conv13_2)
    
    # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
    fire5_bn_mbox_loc = Conv2D(pl_n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', name='fire5_bn_mbox_loc')(fire5_bn)
    fire9_mbox_loc = Conv2D(pl_n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', name='fire9_mbox_loc')(fire9)
    fire10_mbox_loc = Conv2D(pl_n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', name='fire10_mbox_loc')(fire10)
    fire11_mbox_loc = Conv2D(pl_n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', name='fire11_mbox_loc')(fire11)
    conv12_2_mbox_loc = Conv2D(pl_n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', name='conv12_2_mbox_loc')(conv12_2)
    conv13_2_mbox_loc = Conv2D(pl_n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', name='conv13_2_mbox_loc')(conv13_2)

    
    ### Generate the anchor boxes (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)
    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    fire5_bn_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=pl_aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='fire5_bn_mbox_priorbox')(fire5_bn_mbox_loc)
    fire9_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=pl_aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='fire9_mbox_priorbox')(fire9_mbox_loc)
    fire10_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=pl_aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='fire10_mbox_priorbox')(fire10_mbox_loc)
    fire11_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=pl_aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='fire11_mbox_priorbox')(fire11_mbox_loc)
    conv12_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=pl_aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv12_2_mbox_priorbox')(conv12_2_mbox_loc)
    conv13_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=pl_aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv13_2_mbox_priorbox')(conv13_2_mbox_loc)

    ### Reshape
    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    fire5_bn_mbox_conf_reshape = Reshape((-1, n_classes), name='fire5_bn_mbox_conf_reshape')(fire5_bn_mbox_conf)
    fire9_mbox_conf_reshape = Reshape((-1, n_classes), name='fire9_mbox_conf_reshape')(fire9_mbox_conf)
    fire10_mbox_conf_reshape = Reshape((-1, n_classes), name='fire10_mbox_conf_reshape')(fire10_mbox_conf)
    fire11_mbox_conf_reshape = Reshape((-1, n_classes), name='fire11_mbox_conf_reshape')(fire11_mbox_conf)
    conv12_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv12_2_mbox_conf_reshape')(conv12_2_mbox_conf)
    conv13_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv13_2_mbox_conf_reshape')(conv13_2_mbox_conf)
    # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    fire5_bn_mbox_loc_reshape = Reshape((-1, 4), name='fire5_bn_mbox_loc_reshape')(fire5_bn_mbox_loc)
    fire9_mbox_loc_reshape = Reshape((-1, 4), name='fire9_mbox_loc_reshape')(fire9_mbox_loc)
    fire10_mbox_loc_reshape = Reshape((-1, 4), name='fire10_mbox_loc_reshape')(fire10_mbox_loc)
    fire11_mbox_loc_reshape = Reshape((-1, 4), name='fire11_mbox_loc_reshape')(fire11_mbox_loc)
    conv12_2_mbox_loc_reshape = Reshape((-1, 4), name='conv12_2_mbox_loc_reshape')(conv12_2_mbox_loc)
    conv13_2_mbox_loc_reshape = Reshape((-1, 4), name='conv13_2_mbox_loc_reshape')(conv13_2_mbox_loc)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    fire5_bn_mbox_priorbox_reshape = Reshape((-1, 8), name='fire5_bn_mbox_priorbox_reshape')(fire5_bn_mbox_priorbox)
    fire9_mbox_priorbox_reshape = Reshape((-1, 8), name='fire9_mbox_priorbox_reshape')(fire9_mbox_priorbox)
    fire10_mbox_priorbox_reshape = Reshape((-1, 8), name='fire10_mbox_priorbox_reshape')(fire10_mbox_priorbox)
    fire11_mbox_priorbox_reshape = Reshape((-1, 8), name='fire11_mbox_priorbox_reshape')(fire11_mbox_priorbox)
    conv12_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv12_2_mbox_priorbox_reshape')(conv12_2_mbox_priorbox)
    conv13_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv13_2_mbox_priorbox_reshape')(conv13_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([fire5_bn_mbox_conf_reshape,
                                                       fire9_mbox_conf_reshape,
                                                       fire10_mbox_conf_reshape,
                                                       fire11_mbox_conf_reshape,
                                                       conv12_2_mbox_conf_reshape,
                                                       conv13_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([fire5_bn_mbox_loc_reshape,
                                                     fire9_mbox_loc_reshape,
                                                     fire10_mbox_loc_reshape,
                                                     fire11_mbox_loc_reshape,
                                                     conv12_2_mbox_loc_reshape,
                                                     conv13_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([fire5_bn_mbox_priorbox_reshape,
                                                               fire9_mbox_priorbox_reshape,
                                                               fire10_mbox_priorbox_reshape,
                                                               fire11_mbox_priorbox_reshape,
                                                               conv12_2_mbox_priorbox_reshape,
                                                               conv13_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    model = Model(inputs=x, outputs=predictions)

    # Get the spatial dimensions (height, width) of the predictor conv layers, we need them to
    # be able to generate the default boxes for the matching process outside of the model during training.
    # Note that the original implementation performs anchor box matching inside the loss function. We don't do that.
    # Instead, we'll do it in the batch generator function.
    # The spatial dimensions are the same for the confidence and localization predictors, so we just take those of the conf layers.
    predictor_sizes = np.array([fire5_bn_mbox_conf._keras_shape[1:3],
                                 fire9_mbox_conf._keras_shape[1:3],
                                 fire10_mbox_conf._keras_shape[1:3],
                                 fire11_mbox_conf._keras_shape[1:3],
                                 conv12_2_mbox_conf._keras_shape[1:3],
                                 conv13_2_mbox_conf._keras_shape[1:3]])

    return model, predictor_sizes
