# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/20

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from keras.models import Model
from keras.layers import Input, Activation, Conv2D, Lambda, Reshape, Concatenate
from keras.layers import MaxPooling2D, BatchNormalization

from .darknet import DarknetConv2D


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

def squeezenet_body(inputs, num_anchors, num_classes):
    conv01 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal", name="conv01")(inputs)
    pool0 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool0", padding="same")(conv01)
    conv02 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu", kernel_initializer="he_normal", name="conv02")(pool0)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool1", padding="same")(conv02)

    fire2 = _fire(pool1, (16, 64, 64), name="fire2")
    fire3 = _fire(fire2, (16, 64, 64), name="fire3")
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name="pool3", padding="same")(fire3)

    fire4 = _fire(pool3, (32, 128, 128), name="fire4")
    fire5 = _fire(fire4, (32, 128, 128), name="fire5")
    fire6 = _fire(fire5, (48, 192, 192), name="fire6")
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5', padding="same")(fire6)

    fire7 = _fire(pool5, (48, 192, 192), name="fire7")
    fire8 = _fire(fire6, (48, 192, 192), name="fire8")
    fire9 = _fire_with_bn(fire7, (64, 256, 256), name="fire9")
    pool9 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool9', padding="same")(fire9)

    fire10 = _fire_with_bn(pool9, (96, 384, 384), name="fire10")
    fire11 = _fire_with_bn(fire10, (64, 256, 256), name="fire11")
    fire12 = _fire_with_bn(fire11, (48, 192, 192), name="fire12")
    fire13 = _fire_with_bn(fire12, (32, 128, 128), name="fire13")
    outputs = DarknetConv2D(num_anchors * (num_classes + 5), (1, 1))(fire13)
    return Model(inputs, outputs)

