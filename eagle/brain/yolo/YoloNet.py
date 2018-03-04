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


class YoloNet(BaseYoloNet):
    def __init__(self, common_params, net_params, test=False):
        super(YoloNet, self).__init__(common_params, net_params)

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
        conv10 = self.conv2d(
            "conv10", images, kernel_size=[7, 7, 3, 64], stride=2)
        pool10 = self.max_pool(conv10, kernel_size=[2, 2], stride=2)

        conv20 = self.conv2d(
            "conv20", pool10, kernel_size=[3, 3, 64, 192], stride=1)
        pool20 = self.max_pool(conv20, kernel_size=[2, 2], stride=2)

        conv31 = self.conv2d(
            "conv31", pool20, kernel_size=[1, 1, 192, 128], stride=1)
        conv32 = self.conv2d(
            "conv32", conv31, kernel_size=[3, 3, 128, 256], stride=1)
        conv33 = self.conv2d(
            "conv33", conv32, kernel_size=[1, 1, 256, 256], stride=1)
        conv34 = self.conv2d(
            "conv34", conv33, kernel_size=[3, 3, 256, 512], stride=1)
        pool30 = self.max_pool(conv34, kernel_size=[2, 2,], stride=1)

        layer_id = 4
        conv = pool30
        for _ in range(4):
            conv = self.conv2d(
                "conv{}1".format(layer_id),
                input=conv,
                kernel_size=[1, 1, 512, 256],
                stride=1)
            conv = self.conv2d(
                "conv{}2".format(layer_id),
                input=conv,
                kernel_size=[3, 3, 256, 512],
                stride=1)
            layer_id += 1

        conv = self.conv2d(
            "conv{}0".format(layer_id),
            input=conv,
            kernel_size=[1, 1, 512, 512],
            stride=1)
        layer_id += 1

        conv = self.conv2d(
            "conv{}0".format(layer_id),
            input=conv,
            kernel_size=[3, 3, 512, 1024],
            stride=1)
        layer_id += 1

        conv = self.max_pool(conv, kernel_size=[2, 2], stride=2)

        for _ in range(2):
            conv = self.conv2d(
                "conv{}1".format(layer_id),
                input=conv,
                kernel_size=[1, 1, 1024, 512],
                stride=1)
            conv = self.conv2d(
                "conv{}2".format(layer_id),
                input=conv,
                kernel_size=[3, 3, 512, 1024],
                stride=1)
            layer_id += 1

        conv = self.conv2d(
            "conv{}0".format(layer_id),
            input=conv,
            kernel_size=[3, 3, 1024, 1024],
            stride=1)
        layer_id += 1
        conv = self.conv2d(
            "conv{}0".format(layer_id),
            input=conv,
            kernel_size=[3, 3, 1024, 1024],
            stride=2)
        layer_id += 1

        for _ in range(2):
            conv = self.conv2d(
                "conv{}0".format(layer_id),
                input=conv,
                kernel_size=[3, 3, 1024, 1024],
                stride=1)
            layer_id += 1

        # Fully connected layer
        fc1 = self.fc_layer("fc1", conv, in_dim=49 * 1024, out_dim=4096)
        fc1 = tf.nn.dropout(fc1, keep_prob=0.5)
        fc2 = self.fc_layer(
            "fc2",
            fc1,
            in_dim=4096,
            out_dim=self.cell_size ** 2 * (
                self.num_classes + 5 * self.boxes_per_cell),
            leaky=False)
        predicts = tf.reshape(
            fc2,
            shape=[tf.shape(fc2)[0], self.cell_size, self.cell_size,
                   self.num_classes + 5 * self.boxes_per_cell])
        return predicts

    def loss(self, predicts, labels, objects_num):
        """Add Loss to all the trainable variable
        Parameters:
            predicts: 4-D tensor
            [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
            labels: 3-D tensor
            [batch_size, max_objects, 5]
            objects_num: 1-D tensor [batch_size]
        """
        class_loss = tf.constant(0.0, tf.float32)
        object_loss = tf.constant(0.0, tf.float32)
        noobject_loss = tf.constant(0.0, tf.float32)
        coord_loss = tf.constant(0.0, tf.float32)
        loss = [0, 0, 0, 0]

        for i in range(self.batch_size):
            predict = predicts[i, :, :, :]
            label = labels[i, :, :]
            object_num = objects_num[i]
            nilboy = tf.ones([7, 7, 2])
            tuple_results = tf.while_loop(
                cond=self.loss_cond,
                body=self.loss_body,
                loop_vars=[
                    tf.constant(0),
                    object_num,
                    [class_loss, object_loss, noobject_loss, coord_loss],
                    predict,
                    label,
                    nilboy
                ]
            )
            for j in range(len(loss)):
                loss[j] += tuple_results[2][j]
            nilboy = tuple_results[5]

        tf.add_to_collection(
            "losses",
            (loss[0] + loss[1] + loss[2] + loss[3]) / self.batch_size)
        tf.summary.scalar("class_loss", loss[0] / self.batch_size)
        tf.summary.scalar("object_loss", loss[1] / self.batch_size)
        tf.summary.scalar("noobject_loss", loss[2] / self.batch_size)
        tf.summary.scalar("coord_loss", loss[3] / self.batch_size)
        tf.summary.scalar("weight_loss",
                          tf.add_n(tf.get_collection("losses")) -
                          tf.get_collection("losses"))
        return tf.add_n(tf.get_collection("losses"), name="total_loss"), nilboy

    def loss_cond(self, num, object_num, loss, predicts, labels, nilboy):
        pass

    def loss_body(self, num, object_num, loss, predicts, labels, nilboy):
        pass
    