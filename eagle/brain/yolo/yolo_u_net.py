# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from eagle.brain.yolo.net import Net


class YOLOUNet(Net):
    def __init__(self, common_params, net_params, test=False):
        super(YOLOUNet, self).__init__(common_params, net_params)
        # process params
        self.image_size = int(common_params['image_size'])
        self.num_classes = int(common_params['num_classes'])
        self.batch_size = int(common_params['batch_size'])

        self.cell_size = int(net_params['cell_size'])
        self.weight_decay = float(net_params['weight_decay'])
        self.boxes_per_cell = int(net_params['boxes_per_cell'])

        if not test:
            self.object_scale = float(net_params['object_scale'])
            self.noobject_scale = float(net_params['noobject_scale'])
            self.class_scale = float(net_params['class_scale'])
            self.coord_scale = float(net_params['coord_scale'])

    def inference(self, images):
        # (32, 254, 254, 32)
        conv = tf.layers.Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding='valid',
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="conv1"
        )(images)

        # (32, 127, 127, 32)
        conv = tf.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            name="pool1"
        )(conv)

        # (32, 62, 62, 64)
        conv = tf.layers.Conv2D(
            filters=64,
            kernel_size=(5, 5),
            strides=(2, 2),
            padding='valid',
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="conv2"
        )(conv)

        # (32, 31, 31, 64)
        conv = tf.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            name="pool2"
        )(conv)

        # (32, 29, 29, 128)
        conv3 = tf.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="conv3"
        )(conv)

        # (32, 15, 15, 128)
        pool3 = tf.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='same',
            name="pool3"
        )(conv3)

        # (32, 11, 11, 128)
        conv4 = tf.layers.Conv2D(
            filters=128,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='valid',
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="conv4"
        )(pool3)

        # (32, 9, 9, 256)
        conv5 = tf.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="conv5"
        )(conv4)

        # (32, 4, 4, 256)
        pool5 = tf.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            name="pool5"
        )(conv5)

        ## 添加反卷积操作
        # (32, 9, 9, 256)
        dconv5 = tf.layers.Conv2DTranspose(
            filters=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="dconv5"
        )(pool5)

        # (32, 11, 11, 128)
        dconv4 = tf.layers.Conv2DTranspose(
            filters=128,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="dconv4"
        )(dconv5)

        # (32, 11, 11, 256)
        hyper_dcov4 = tf.concat(
            values=[conv4, dconv4], axis=3, name="hyper_dcov4")

        # (32, 15, 15, 192)
        dpool3 = tf.layers.Conv2DTranspose(
            filters=192,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="valid",
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="dpool3"
        )(hyper_dcov4)

        print(dpool3)

    def loss(self, predicts, labels, objects_num):
        pass
