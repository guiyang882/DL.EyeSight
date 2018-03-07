# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Conv2D, Concatenate
from keras.layers import BatchNormalization

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
