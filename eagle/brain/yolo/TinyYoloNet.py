# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from eagle.brain.yolo.BaseYoloNet import BaseYoloNet


class TinyYoloNet(BaseYoloNet):
    def __init__(self, common_params, net_params, test=False):
        super(TinyYoloNet, self).__init__(common_params, net_params)

        if not test:
            self.object_scale = float(net_params["object_scale"])
            self.noobject_scale = float(net_params["noobject_scale"])
            self.class_scale = float(net_params["class_scale"])
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
            self.num_classes + self.boxes_per_cell * 5), leaky=False)

        n1 = self.cell_size ** 2 * self.num_classes
        n2 = n1 + self.cell_size ** 2 * self.boxes_per_cell

        class_probs = tf.reshape(
            fc[:, 0:n1],
            shape=(-1, self.cell_size, self.cell_size, self.num_classes))
        scales = tf.reshape(
            fc[:, n1:n2],
            shape=(-1, self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = tf.reshape(
            fc[:, n2:],
            shape=(-1, self.cell_size, self.cell_size, self.boxes_per_cell * 4))

        predicts = tf.concat([class_probs, scales, boxes], axis=3)
        return predicts

    def loss(self, predicts, labels, objects_num):
        """Add loss to all the trainable variable
        Parameters:
            predicts: 4-D tensor
            [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
            labels: 3-D tensor [batch_size, max_objects, 5]
            objects_num: 1-D tensor [batch_size]
        """
        class_loss = tf.constant(0.0, tf.float32)
        object_loss = tf.constant(0.0, tf.float32)
        noobject_loss = tf.constant(0.0, tf.float32)
        coord_loss = tf.constant(0.0, tf.float32)
        loss = [0] * 4

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
        tf.summary.scalar(
            "weight_loss",
            tf.add_n(tf.get_collection("losses")) -
            ((loss[0] + loss[1] + loss[2] + loss[3]) / self.batch_size))

        return tf.add_n(tf.get_collection("losses"), name="total_loss"), nilboy

    def loss_cond(self, num, object_num, loss, predicts, labels, nilboy):
        return num < object_num

    def loss_body(self, num, object_num, loss, predicts, labels, nilboy):
        """Calculate loss
        Parameters:
            predicts: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
            labels: [max_objects, 5] (x_center, y_center, w, h, class)
        """
        label = labels[num:num+1, :]
        label = tf.reshape(label, [-1])

        # calculate objects tensor [cell_size, cell_size]
        min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
        max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)
        min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
        max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)

        min_x, min_y = tf.floor(min_x), tf.floor(min_y)
        max_x, max_y = tf.ceil(max_x), tf.ceil(max_y)

        objects = tf.ones(
            shape=tf.cast(
                tf.stack([max_y - min_y, max_x - min_x]),
                dtype=tf.int32),
            dtype=tf.float32)
        paddings = tf.cast(
            tf.stack(
                [min_y, self.cell_size - max_y, min_x, self.cell_size - max_x]),
            dtype=tf.int32)
        paddings = tf.reshape(paddings, shape=(2, 2))
        objects = tf.pad(objects, paddings, "CONSTANT")

        # calculate responsible tensor [cell_size, cell_size]
        center_x = label[0] / (self.image_size / self.cell_size)
        center_x = tf.floor(center_x)
        center_y = label[1] / (self.image_size / self.cell_size)
        center_y = tf.floor(center_y)
        response = tf.ones([1, 1], tf.float32)
        paddings = tf.cast(
            tf.stack(
                [center_y, self.cell_size - center_y - 1,
                 center_x, self.cell_size - center_x - 1]),
            dtype=tf.int32)
        paddings = tf.reshape(paddings, shape=(2, 2))
        response = tf.pad(response, paddings, "CONSTANT")

        # calculate iou_predict_truth [cell_size, cell_size, boxes_per_cell]
        predict_boxes = predicts[:, :, self.num_classes + self.boxes_per_cell:]
        predict_boxes = tf.reshape(
            predict_boxes,
            shape=[self.cell_size, self.cell_size, self.boxes_per_cell, 4])
        predict_boxes = predict_boxes * [self.image_size / self.cell_size,
                                         self.image_size / self.cell_size,
                                         self.image_size,
                                         self.image_size]
        base_boxes = np.zeros([self.cell_size, self.cell_size, 4])
        for y in range(self.cell_size):
            for x in range(self.cell_size):
                base_boxes[y, x, :] = [
                    self.image_size / self.cell_size * x,
                    self.image_size / self.cell_size * y,
                    0,
                    0]
        base_boxes = np.tile(
            np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]),
            [1, 1, self.boxes_per_cell, 1])
        predict_boxes = base_boxes + predict_boxes
        iou_predict_truth = self.iou(predict_boxes, label[0:4])

        # calculate C [cell_size, cell_size, boxes_per_cell]
        C = iou_predict_truth * tf.reshape(
            response, [self.cell_size, self.cell_size, 1])

        # calculate I [cell_size, cell_size, boxes_per_cell]
        I = iou_predict_truth * tf.reshape(
            response, [self.cell_size, self.cell_size, 1])
        max_I = tf.reduce_max(I, axis=2, keep_dims=True)
        I = tf.cast((I >= max_I), tf.float32) * tf.reshape(
            response, [self.cell_size, self.cell_size, 1])

        # calculate no_I [cell_size, cell_size, boxes_per_cell]
        no_I = tf.ones_like(I, dtype=tf.float32) - I
        p_C = predicts[:, :, self.num_classes:self.num_classes+self.boxes_per_cell]

        # calculate truth x, y, sqrt_w, sqrt_h
        x, y = label[0], label[1]
        sqrt_w = tf.sqrt(tf.abs(label[2]))
        sqrt_h = tf.sqrt(tf.abs(label[3]))

        # calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h
        p_x = predict_boxes[:, :, :, 0]
        p_y = predict_boxes[:, :, :, 1]
        p_sqrt_w = tf.sqrt(
            tf.minimum(
                self.image_size * 1.0,
                tf.maximum(0.0, predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(
            tf.minimum(
                self.image_size * 1.0,
                tf.maximum(0.0, predict_boxes[:, :, :, 3])))

        # calculate truth p [num_classes]
        P = tf.one_hot(
            tf.cast(label[4], dtype=tf.int32),
            depth=self.num_classes,
            dtype=tf.float32)
        # calculate predict p_P [cell_size, cell_size, num_classes]
        p_P = predicts[:, :, 0:self.num_classes]

        # class_loss
        class_loss = self.class_scale * tf.nn.l2_loss(
            tf.reshape(objects, [self.cell_size, self.cell_size, 1]) * (p_P - P))

        # object_loss
        object_loss = self.object_scale * tf.nn.l2_loss(I * (p_C - C))

        # noobject_loss
        noobject_loss = self.noobject_scale * tf.nn.l2_loss(no_I * p_C)

        # coord_loss
        coord_loss = self.coord_scale * (
            tf.nn.l2_loss(I * (p_x - x) / (self.image_size / self.cell_size)) +
            tf.nn.l2_loss(I * (p_y - y) / (self.image_size / self.cell_size)) +
            tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w)) / self.image_size +
            tf.nn.l2_loss(I * (p_sqrt_h - sqrt_h)) / self.image_size
        )

        nilboy = I

        loss = [loss[0] + class_loss, loss[1] + object_loss,
                loss[2] + noobject_loss, loss[3] + coord_loss]

        return num + 1, object_num, loss, predicts, labels, nilboy
