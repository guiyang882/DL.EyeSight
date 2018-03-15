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

    def set_cell_size(self, grid_size):
        self.cell_size = grid_size

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

        # create hyper feature map
        hyper_grid_15 = tf.concat(
            values=[pool3, dpool3], axis=3, name="hyper_grid_15")
        hyper_grid_9 = tf.concat(
            values=[conv5, dconv5], axis=3, name="hyper_grid_9")

        # create the fc layer for every feature map for grid cell == 9
        hyper_grid_9 = tf.reshape(hyper_grid_9, shape=(self.batch_size, -1))
        fc1_g9 = tf.contrib.layers.fully_connected(
            inputs=hyper_grid_9,
            num_outputs=256,
            activation_fn=tf.nn.leaky_relu
        )

        fc2_g9 = tf.contrib.layers.fully_connected(
            inputs=fc1_g9,
            num_outputs=1024,
            activation_fn=tf.nn.leaky_relu
        )

        out_dim = 9 * 9 * (self.num_classes + self.boxes_per_cell * 5)
        fc3_g9 = tf.contrib.layers.fully_connected(
            inputs=fc2_g9,
            num_outputs=out_dim,
            activation_fn=tf.nn.relu
        )

        n1 = 9 * 9 * self.num_classes
        n2 = n1 + 9 * 9 * self.boxes_per_cell

        class_probs_g9 = tf.reshape(fc3_g9[:, 0:n1], (
            -1, 9, 9, self.num_classes))
        scales_g9 = tf.reshape(fc3_g9[:, n1:n2], (
            -1, 9, 9, self.boxes_per_cell))
        boxes_g9 = tf.reshape(fc3_g9[:, n2:], (
            -1, 9, 9, self.boxes_per_cell * 4))

        predicts_g9 = tf.concat([class_probs_g9, scales_g9, boxes_g9], 3)

        # create the fc layer for every feature map for grid cell == 15
        hyper_grid_15 = tf.reshape(hyper_grid_15, shape=(self.batch_size, -1))
        fc1_g15 = tf.contrib.layers.fully_connected(
            inputs=hyper_grid_15,
            num_outputs=256,
            activation_fn=tf.nn.leaky_relu
        )

        fc2_g15 = tf.contrib.layers.fully_connected(
            inputs=fc1_g15,
            num_outputs=1024,
            activation_fn=tf.nn.leaky_relu
        )

        out_dim = 15 * 15 * (self.num_classes + self.boxes_per_cell * 5)
        fc3_g15 = tf.contrib.layers.fully_connected(
            inputs=fc2_g15,
            num_outputs=out_dim,
            activation_fn=tf.nn.relu
        )

        n1 = 15 * 15 * self.num_classes
        n2 = n1 + 15 * 15 * self.boxes_per_cell

        class_probs_g15 = tf.reshape(fc3_g15[:, 0:n1], (
            -1, 15, 15, self.num_classes))
        scales_g15 = tf.reshape(fc3_g15[:, n1:n2], (
            -1, 15, 15, self.boxes_per_cell))
        boxes_g15 = tf.reshape(fc3_g15[:, n2:], (
            -1, 15, 15, self.boxes_per_cell * 4))

        predicts_g15 = tf.concat([class_probs_g15, scales_g15, boxes_g15], 3)

        res = {
            "predicts_g9": predicts_g9,
            "predicts_g15": predicts_g15
        }
        return res

    def loss(self, predicts, labels, objects_num):
        """Add Loss to all the trainable variables

                Args:
                  predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
                  ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
                  labels  : 3-D tensor of [batch_size, max_objects, 5]
                  objects_num: 1-D tensor [batch_size]
                """
        class_loss = tf.constant(0, tf.float32)
        object_loss = tf.constant(0, tf.float32)
        noobject_loss = tf.constant(0, tf.float32)
        coord_loss = tf.constant(0, tf.float32)
        loss = [0, 0, 0, 0]
        for i in range(self.batch_size):
            predict = predicts[i, :, :, :]
            label = labels[i, :, :]
            object_num = objects_num[i]
            nilboy = tf.ones([self.cell_size, self.cell_size, 2])
            tuple_results = tf.while_loop(
                self.loss_cond,
                self.loss_body,
                [tf.constant(0),
                 object_num,
                 [class_loss, object_loss, noobject_loss, coord_loss],
                 predict,
                 label,
                 nilboy])
            for j in range(4):
                loss[j] = loss[j] + tuple_results[2][j]
            nilboy = tuple_results[5]

        tf.add_to_collection('losses', (
            loss[0] + loss[1] + loss[2] + loss[3]) / self.batch_size)

        tf.summary.scalar('class_loss', loss[0] / self.batch_size)
        tf.summary.scalar('object_loss', loss[1] / self.batch_size)
        tf.summary.scalar('noobject_loss', loss[2] / self.batch_size)
        tf.summary.scalar('coord_loss', loss[3] / self.batch_size)
        tf.summary.scalar('weight_loss',
                          tf.add_n(tf.get_collection('losses')) - (
                              loss[0] + loss[1] + loss[2] + loss[
                                  3]) / self.batch_size)

        return tf.add_n(tf.get_collection('losses'), name='total_loss'), nilboy

    def loss_cond(self, num, object_num, loss, predict, label, nilboy):
        return num < object_num

    def loss_body(self, num, object_num, loss, predict, labels, nilboy):
        """
        calculate loss
        Args:
          predict: 3-D tensor [cell_size, cell_size, 5 * boxes_per_cell]
          labels : [max_objects, 5]  (x_center, y_center, w, h, class)
        """
        label = labels[num:num + 1, :]
        label = tf.reshape(label, [-1])

        # calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        min_x = (label[0] - label[2] / 2) / (self.image_size / self.cell_size)
        max_x = (label[0] + label[2] / 2) / (self.image_size / self.cell_size)

        min_y = (label[1] - label[3] / 2) / (self.image_size / self.cell_size)
        max_y = (label[1] + label[3] / 2) / (self.image_size / self.cell_size)

        min_x = tf.floor(min_x)
        min_y = tf.floor(min_y)

        max_x = tf.ceil(max_x)
        max_y = tf.ceil(max_y)

        temp = tf.cast(tf.stack([max_y - min_y, max_x - min_x]), dtype=tf.int32)
        objects = tf.ones(temp, tf.float32)

        temp = tf.cast(tf.stack(
            [min_y, self.cell_size - max_y, min_x, self.cell_size - max_x]),
                       tf.int32)
        temp = tf.reshape(temp, (2, 2))
        objects = tf.pad(objects, temp, "CONSTANT")

        # calculate objects  tensor [CELL_SIZE, CELL_SIZE]
        # calculate responsible tensor [CELL_SIZE, CELL_SIZE]
        center_x = label[0] / (self.image_size / self.cell_size)
        center_x = tf.floor(center_x)

        center_y = label[1] / (self.image_size / self.cell_size)
        center_y = tf.floor(center_y)

        response = tf.ones([1, 1], tf.float32)

        temp = tf.cast(tf.stack(
            [center_y, self.cell_size - center_y - 1, center_x,
             self.cell_size - center_x - 1]), tf.int32)
        temp = tf.reshape(temp, (2, 2))
        response = tf.pad(response, temp, "CONSTANT")
        # objects = response

        # calculate iou_predict_truth [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        predict_boxes = predict[:, :, self.num_classes + self.boxes_per_cell:]

        predict_boxes = tf.reshape(predict_boxes,
                                   [self.cell_size, self.cell_size,
                                    self.boxes_per_cell, 4])

        predict_boxes = predict_boxes * [self.image_size / self.cell_size,
                                         self.image_size / self.cell_size,
                                         self.image_size, self.image_size]

        base_boxes = np.zeros([self.cell_size, self.cell_size, 4])

        for y in range(self.cell_size):
            for x in range(self.cell_size):
                base_boxes[y, x, :] = [self.image_size / self.cell_size * x,
                                       self.image_size / self.cell_size * y,
                                       0, 0]
        base_boxes = np.tile(
            np.resize(base_boxes, [self.cell_size, self.cell_size, 1, 4]),
            [1, 1, self.boxes_per_cell, 1])

        predict_boxes = base_boxes + predict_boxes

        iou_predict_truth = self.iou(predict_boxes, label[0:4])
        # calculate C [cell_size, cell_size, boxes_per_cell]
        C = iou_predict_truth * tf.reshape(response,
                                           [self.cell_size, self.cell_size, 1])

        # calculate I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        I = iou_predict_truth * tf.reshape(response,
                                           (self.cell_size, self.cell_size, 1))

        max_I = tf.reduce_max(I, 2, keep_dims=True)

        I = tf.cast((I >= max_I), tf.float32) * tf.reshape(response, (
        self.cell_size, self.cell_size, 1))

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        no_I = tf.ones_like(I, dtype=tf.float32) - I

        p_C = predict[:, :,
              self.num_classes:self.num_classes + self.boxes_per_cell]

        # calculate truth x,y,sqrt_w,sqrt_h 0-D
        x = label[0]
        y = label[1]

        sqrt_w = tf.sqrt(tf.abs(label[2]))
        sqrt_h = tf.sqrt(tf.abs(label[3]))

        # calculate predict p_x, p_y, p_sqrt_w, p_sqrt_h 3-D [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        p_x = predict_boxes[:, :, :, 0]
        p_y = predict_boxes[:, :, :, 1]

        p_sqrt_w = tf.sqrt(
            tf.minimum(
                self.image_size * 1.0,
                tf.maximum(0.0,
                           predict_boxes[:, :, :, 2])))
        p_sqrt_h = tf.sqrt(
            tf.minimum(
                self.image_size * 1.0,
                tf.maximum(0.0,
                           predict_boxes[:, :, :, 3])))

        # calculate truth p 1-D tensor [NUM_CLASSES]
        P = tf.one_hot(tf.cast(label[4], tf.int32),
                       depth=self.num_classes,
                       dtype=tf.float32)

        # calculate predict p_P 3-D tensor [CELL_SIZE, CELL_SIZE, NUM_CLASSES]
        p_P = predict[:, :, 0:self.num_classes]

        # class_loss
        class_loss = tf.nn.l2_loss(
            tf.reshape(objects, (self.cell_size, self.cell_size, 1)) * (
            p_P - P)) * self.class_scale

        # object_loss
        object_loss = tf.nn.l2_loss(I * (p_C - C)) * self.object_scale

        # noobject_loss
        noobject_loss = tf.nn.l2_loss(no_I * (p_C)) * self.noobject_scale

        # coord_loss
        coord_loss = (tf.nn.l2_loss(
            I * (p_x - x) / (self.image_size / self.cell_size)) +
                      tf.nn.l2_loss(
                          I * (p_y - y) / (self.image_size / self.cell_size)) +
                      tf.nn.l2_loss(I * (p_sqrt_w - sqrt_w)) / self.image_size +
                      tf.nn.l2_loss(I * (
                      p_sqrt_h - sqrt_h)) / self.image_size) * self.coord_scale

        nilboy = I

        return num + 1, object_num, [loss[0] + class_loss,
                                     loss[1] + object_loss,
                                     loss[2] + noobject_loss,
                                     loss[3] + coord_loss], predict, labels, nilboy

    def iou(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        boxes1 = tf.stack([boxes1[:, :, :, 0] - boxes1[:, :, :, 2] / 2,
                           boxes1[:, :, :, 1] - boxes1[:, :, :, 3] / 2,
                           boxes1[:, :, :, 0] + boxes1[:, :, :, 2] / 2,
                           boxes1[:, :, :, 1] + boxes1[:, :, :, 3] / 2])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 0])
        boxes2 = tf.stack([boxes2[0] - boxes2[2] / 2, boxes2[1] - boxes2[3] / 2,
                           boxes2[0] + boxes2[2] / 2,
                           boxes2[1] + boxes2[3] / 2])

        # calculate the left up point
        lu = tf.maximum(boxes1[:, :, :, 0:2], boxes2[0:2])
        rd = tf.minimum(boxes1[:, :, :, 2:], boxes2[2:])

        # intersection
        intersection = rd - lu

        inter_square = intersection[:, :, :, 0] * intersection[:, :, :, 1]

        mask = tf.cast(intersection[:, :, :, 0] > 0, tf.float32) * tf.cast(
            intersection[:, :, :, 1] > 0, tf.float32)

        inter_square = mask * inter_square

        # calculate the boxs1 square and boxs2 square
        square1 = (boxes1[:, :, :, 2] - boxes1[:, :, :, 0]) * (
        boxes1[:, :, :, 3] - boxes1[:, :, :, 1])
        square2 = (boxes2[2] - boxes2[0]) * (boxes2[3] - boxes2[1])

        return inter_square / (square1 + square2 - inter_square + 1e-6)
