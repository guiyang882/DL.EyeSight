# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from abc import ABCMeta, abstractmethod


class BaseYoloNet(object):
    __metaclass__ = ABCMeta

    def __init__(self, common_params, net_params):
        if not isinstance(common_params, dict):
            raise ValueError("common_params' type should be dict")
        if not isinstance(net_params, dict):
            raise ValueError("net_params' type should be dict")

        # extract the parameters
        self.image_size = int(common_params["image_size"])
        self.num_classes = int(common_params["num_classes"])
        self.batch_size = int(common_params["batch_size"])

        self.cell_size = int(net_params["cell_size"])
        self.boxes_per_cell = int(net_params["boxes_per_cell"])
        self.weight_decay = float(net_params["weight_decay"])

        self.pretrained_collection = list()
        self.trainable_collection = list()

    def _variable_on_cpu(self, name, shape,
                         initializer, pretrain=True, train=True):
        with tf.device("/cpu:0"):
            var = tf.get_variable(name, shape,
                                  initializer=initializer, dtype=tf.float32)
            if pretrain:
                self.pretrained_collection.append(var)
            if train:
                self.trainable_collection.append(var)
        return var

    def _variable_with_weight_decay(self, name, shape,
                                    stddev, wd, pretrain=True, train=True):
        var = self._variable_on_cpu(
            name,
            shape,
            tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32),
            pretrain=pretrain,
            train=train
        )
        if wd is not None:
            weight_decay = tf.multiply(
                x=tf.nn.l2_loss(var),
                y=wd,
                name="weight_loss"
            )
            tf.add_to_collection("losses", weight_decay)
        return var

    def conv2d(self, scope, input, kernel_size,
               stride=1, pretrain=True, train=True):
        with tf.variable_scope(scope) as scope:
            kernel = self._variable_with_weight_decay(
                name="weights",
                shape=kernel_size,
                stddev=5e-2,
                wd=self.weight_decay,
                pretrain=pretrain,
                train=train
            )
            conv = tf.nn.conv2d(
                input, kernel, [1, stride, stride, 1], padding='SAME')
            biases = self._variable_on_cpu(
                "biases", kernel_size[3:], tf.constant_initializer(0.0),
                pretrain=pretrain, train=train)
            res = tf.nn.bias_add(conv, biases)
            res = self.leaky_relu(res)
        return res

    def leaky_relu(self, x, alpha=0.1, dtype=tf.float32):
        """formula
        if x > 0:
            return x
        else:
            return alpha * x
        """
        x = tf.cast(x, dtype=dtype)
        bool_mask = x > 0
        mask = tf.cast(bool_mask, dtype=dtype)
        return 1.0 * mask * x + alpha * (1 - mask) * x

    def max_pool(self, input, kernel_size, stride):
        res = tf.nn.max_pool(input,
                             ksize=[1, kernel_size[0], kernel_size[1], 1],
                             strides=[1, stride, stride, 1],
                             padding="SAME")
        return res

    def fc_layer(self, scope, input, in_dim, out_dim,
                 leaky=True, pretrain=True, train=True):
        with tf.variable_scope(scope) as scope:
            reshape = tf.reshape(input, [tf.shape(input)[0], -1])
            weights = self._variable_with_weight_decay(
                name="weights",
                shape=[in_dim, out_dim],
                stddev=0.04,
                wd=self.weight_decay,
                pretrain=True,
                train=train
            )
            biases = self._variable_on_cpu(
                name="biases",
                shape=[out_dim],
                initializer=tf.constant_initializer(0.0),
                pretrain=pretrain,
                train=train
            )
            res = tf.matmul(reshape, weights) + biases

            if leaky:
                res = self.leaky_relu(res)
            else:
                res = tf.identity(res, name=scope.name)
        return res

    def iou(self, boxes1, boxes2):
        """calculate ious
        Parameters:
            boxes1: 4-D tensor [cell_size, cell_size, boxes_per_cell, 4]
                ===> (x_center, y_center, w, h)
            boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
        Returns:
            iou: 3-D tensor [cell_size, cell_size, boxes_per_cell]
        """
        boxes1 = tf.stack([
            boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2,
            boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
            boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2,
            boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2
        ])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
        boxes2 = tf.stack([
            boxes2[0] - boxes2[2] / 2,
            boxes2[1] - boxes2[3] / 2,
            boxes2[0] + boxes2[2] / 2,
            boxes2[1] + boxes2[3] / 2
        ])

        # calculate the left up point
        lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

        # intersection
        intersection = rd - lu
        inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

        # find the legal rect in the intersection boxes
        mask = tf.cast(intersection[:, :, :, 0] > 0, dtype=tf.float32) * \
               tf.cast(intersection[:, :, :, 1] > 0, dtype=tf.float32)
        inter_square = mask * inter_square

        # calculate the boxes1 square and boxes2 square
        square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * \
                  (boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        # calculate the iou and return
        return inter_square / (square1 + square2 - inter_square + 1e-6)

    @abstractmethod
    def inference(self, images, params=None):
        """Build the yolo model.
        Parameters:
            images: [batch_sizes, height, width, channel]
        Returns:
            [batch_sizes, cell_size, cell_size, num_clases + 5 * boxes_per_cell]
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, predicts, labels, objects_num):
        """Add Loss to all the trainable variables
        Parameters:
            predicts: [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
                ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
            labels: [batch_size, max_objects, 5]
            objects_num: [batch_size]
        """
        raise NotImplementedError
