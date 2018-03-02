# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from eagle.brain.yolo.BaseYoloNet import BaseYoloNet


class TinyYoloNet(BaseYoloNet):
    def __init__(self, common_params, net_params, test=False):
        super(TinyYoloNet, self).__init__(common_params, net_params)

        if not test:
            self.object_scale = float(net_params["object_scale"])
            self.noobject_scale = float(net_params["noobject_scale"])
            self.calss_scale = float(net_params["class_scale"])
            self.coord_scale = float(net_params["coord_scale"])

    def inference(self, images, params=None):
        """Build the yolo model
        Returns:
            predicts: 4-D tensor
            [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        """
        conv = self.conv2d(
            "conv1", images, kernel_size=[3, 3, 3, 16], stride=1)
        pool = self.max_pool(conv, kernel_size=[2, 2], stride=2)

        conv = self.conv2d(
            "conv2", pool, kernel_size=[3, 3, 16, 32], stride=1)
        pool = self.max_pool(conv, kernel_size=[2, 2], stride=2)

        conv = self.conv2d(
            "conv3", pool, kernel_size=[3, 3, 32, 64], stride=1)
        pool = self.max_pool(conv, kernel_size=[2, 2], stride=2)

        conv = self.conv2d(
            "conv4", pool, kernel_size=[3, 3, 64, 128], stride=1)
        pool = self.max_pool(conv, kernel_size=[2, 2], stride=2)

        conv = self.conv2d(
            "conv5", pool, kernel_size=[3, 3, 128, 256], stride=1)
        pool = self.max_pool(conv, kernel_size=[2, 2], stride=2)

        conv = self.conv2d(
            "conv6", pool, kernel_size=[3, 3, 256, 512], stride=1)
        pool = self.max_pool(conv, kernel_size=[2, 2], stride=2)

        conv = self.conv2d(
            "conv7", pool, kernel_size=[3, 3, 512, 1024], stride=1)
        conv = self.conv2d(
            "conv8", conv, kernel_size=[3, 3, 1024, 1024], stride=1)
        conv = self.conv2d(
            "conv9", conv, kernel_size=[3, 3, 1024, 1024], stride=1)

        conv = tf.transpose(conv, (0, 3, 1, 2))

        # fully connected layer
        fc = self.fc_layer("fc1", conv, self.cell_size ** 2 * 1024, 256)
        fc = self.fc_layer("fc2", fc, 256, 4096)
        fc = self.fc_layer("fc3", fc, 4096, self.cell_size ** 2 * (
            self.num_calsses + self.boxes_per_cell * 5), leaky=False)

        n1 = self.cell_size ** 2 * self.num_calsses
        n2 = n1 + self.cell_size ** 2 * self.boxes_per_cell

        class_probs = tf.reshape(
            fc[:, 0:n1],
            shape=(-1, self.cell_size, self.cell_size, self.num_calsses))
        scales = tf.reshape(
            fc[:, n1:n2],
            shape=(-1, self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = tf.reshape(
            fc[:, n2:],
            shape=(-1, self.cell_size, self.cell_size, self.boxes_per_cell * 4))

        predicts = tf.concat([class_probs, scales, boxes], axis=3)
        return predicts

    def loss(self, predicts, labels, objects_num):
        pass
