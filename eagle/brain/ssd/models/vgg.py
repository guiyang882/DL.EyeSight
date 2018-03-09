# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/18

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from eagle.brain.ssd.loss import Loss
from eagle.brain.ssd.models.net import Net
from eagle.brain.ssd.anchor_boxes import AnchorBoxes


class SSDVGG(Net):
    def __init__(self, common_params, net_params, box_encoder_params):
        super(SSDVGG, self).__init__(common_params, net_params)
        ## 解析common_params
        self.image_size = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])
        self.image_width = int(common_params['image_width'])
        self.image_height = int(common_params['image_height'])
        self.image_channel = int(common_params['image_channel'])
        self.num_classes = int(common_params['num_classes']) + 1

        ## 解析net_params
        self.n_neg_min = int(net_params["n_neg_min"])
        self.loss_alpha = float(net_params["loss_alpha"])
        self.neg_pos_ratio = int(net_params["neg_pos_ratio"])

        ## 解析box_encoder_params
        # 处理predictor_sizes
        predictor_sizes = box_encoder_params["predictor_sizes"]
        items = ""
        for c in predictor_sizes:
            if c.isdigit() or c == ',':
                items += c
        items = list(map(int, items.strip().split(",")))
        predictor_sizes = np.asarray(items, dtype=np.int32).reshape([-1, 2])
        self.predictor_sizes = predictor_sizes

        # 处理scales
        items = box_encoder_params["scales"].strip()[1:-1].split(",")
        scales = list(map(float, items))
        self.scales = scales

        # 处理aspect_ratios_per_layer，其中没一个cell的size不一定一样
        items = box_encoder_params["aspect_ratios_per_layer"].strip()[1:-1]
        tmp = ""
        seq_stack = list()
        aspect_ratios_per_layer = list()
        for c in items:
            if c == '[':
                seq_stack.append(c)
            elif c == ']':
                if len(tmp):
                    aspect_ratios_per_layer.append(
                        list(map(float, tmp.strip(',').split(","))))
                seq_stack.pop()
                tmp = ""
            elif c.isdigit() or c == '.' or c == ',':
                tmp += c
            else:
                pass
        self.aspect_ratios_per_layer = aspect_ratios_per_layer

        two_boxes_for_ar1 = True if box_encoder_params["two_boxes_for_ar1"] \
                                    == "True" else False
        self.two_boxes_for_ar1 = two_boxes_for_ar1

        # 处理variances
        items = box_encoder_params["variances"].strip()[1:-1].split(",")
        variances = list(map(float, items))
        self.variances = np.array(variances, dtype=np.float32)

        coords = box_encoder_params["coords"]
        self.coords = coords

        normalize_coords = True if box_encoder_params["normalize_coords"] == \
                                   "True" else False
        self.normalize_coords = normalize_coords

        pos_iou_threshold = float(box_encoder_params["pos_iou_threshold"])
        neg_iou_threshold = float(box_encoder_params["neg_iou_threshold"])
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold

        self.check_valid()

        ## 现在要先创造出loss的损失函数的管理对象
        self.model_loss_obj = None

    def check_valid(self):
        # 检测参数输入是否在正确
        if len(self.scales) != self.predictor_sizes.shape[0] + 1:
            raise ValueError("len(self.scales) != self.predictor_sizes.shape[0] + 1")

        if len(self.scales) != len(self.aspect_ratios_per_layer) + 1:
            raise ValueError("len(self.scales) != len(self.aspect_ratios_per_layer) + 1")

        if len(self.variances) != 4:
            raise ValueError("len(self.variances) != 4")

        if np.any(self.variances <= 0):
            raise ValueError("np.any(self.variances <= 0)")

        if self.neg_iou_threshold > self.pos_iou_threshold:
            raise ValueError(
                "It cannot be `neg_iou_threshold > pos_iou_threshold`.")

        if not (self.coords == 'minmax' or self.coords == 'centroids'):
            raise ValueError(
                "Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

    def inference(self, images):
        # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
        aspect_ratios_conv4_3 = self.aspect_ratios_per_layer[0]
        aspect_ratios_fc7 = self.aspect_ratios_per_layer[1]
        aspect_ratios_conv6_2 = self.aspect_ratios_per_layer[2]
        aspect_ratios_conv7_2 = self.aspect_ratios_per_layer[3]
        aspect_ratios_conv8_2 = self.aspect_ratios_per_layer[4]
        aspect_ratios_conv9_2 = self.aspect_ratios_per_layer[5]

        # Compute the number of boxes to be predicted per cell for each predictor layer.
        # We need this so that we know how many channels the predictor layers need to have.
        n_boxes = []
        for aspect_ratios in self.aspect_ratios_per_layer:
            # +1 for the second box for aspect ratio 1
            if (1 in aspect_ratios) & self.two_boxes_for_ar1:
                n_boxes.append(len(aspect_ratios) + 1)
            else:
                n_boxes.append(len(aspect_ratios))
        # 4 boxes per cell for the original implementation
        n_boxes_conv4_3 = n_boxes[0]
        # 6 boxes per cell for the original implementation
        n_boxes_fc7 = n_boxes[1]
        # 6 boxes per cell for the original implementation
        n_boxes_conv6_2 = n_boxes[2]
        # 6 boxes per cell for the original implementation
        n_boxes_conv7_2 = n_boxes[3]
        # 4 boxes per cell for the original implementation
        n_boxes_conv8_2 = n_boxes[4]
        # 4 boxes per cell for the original implementation
        n_boxes_conv9_2 = n_boxes[5]

        ### Design the actual network
        conv1_1 = tf.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="conv1_1"
        )(images)

        conv1_2 = tf.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="conv1_2"
        )(conv1_1)

        pool1 = tf.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            name='pool1'
        )(conv1_2)

        conv2_1 = tf.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name="conv2_1"
        )(pool1)

        conv2_2 = tf.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv2_2'
        )(conv2_1)

        pool2 = tf.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            name='pool2'
        )(conv2_2)

        conv3_1 = tf.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv3_1'
        )(pool2)

        conv3_2 = tf.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv3_2'
        )(conv3_1)

        conv3_3 = tf.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv3_3'
        )(conv3_2)

        pool3 = tf.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            name='pool3'
        )(conv3_3)

        conv4_1 = tf.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv4_1'
        )(pool3)

        conv4_2 = tf.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv4_2'
        )(conv4_1)

        conv4_3 = tf.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv4_3'
        )(conv4_2)

        pool4 = tf.layers.MaxPooling2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            name='pool4'
        )(conv4_3)

        conv5_1 = tf.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv5_1'
        )(pool4)

        conv5_2 = tf.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv5_2'
        )(conv5_1)

        conv5_3 = tf.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv5_3'
        )(conv5_2)

        pool5 = tf.layers.MaxPooling2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding='same',
            name='pool5'
        )(conv5_3)

        fc6 = tf.layers.Conv2D(
            filters=1024,
            kernel_size=(3, 3),
            dilation_rate=(6, 6),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='fc6'
        )(pool5)

        fc7 = tf.layers.Conv2D(
            filters=1024,
            kernel_size=(1, 1),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='fc7'
        )(fc6)

        conv6_1 = tf.layers.Conv2D(
            filters=256,
            kernel_size=(1, 1),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv6_1'
        )(fc7)

        conv6_2 = tf.layers.Conv2D(
            filters=512,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv6_2'
        )(conv6_1)

        conv7_1 = tf.layers.Conv2D(
            filters=128,
            kernel_size=(1, 1),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv7_1'
        )(conv6_2)

        conv7_2 = tf.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv7_2'
        )(conv7_1)

        conv8_1 = tf.layers.Conv2D(
            filters=128,
            kernel_size=(1, 1),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv8_1'
        )(conv7_2)

        conv8_2 = tf.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=tf.nn.relu,
            padding='valid',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv8_2'
        )(conv8_1)

        conv9_1 = tf.layers.Conv2D(
            filters=128,
            kernel_size=(1, 1),
            activation=tf.nn.relu,
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv9_1'
        )(conv8_2)

        conv9_2 = tf.layers.Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation=tf.nn.relu,
            padding='valid',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv9_2'
        )(conv9_1)

        # Feed conv4_3 into the L2 normalization layer
        conv4_3_norm = tf.nn.l2_normalize(conv4_3, dim=3, name="conv4_3_norm")
        # conv4_3_norm = L2Normalization(gamma_init=20,
        #                                name='conv4_3_norm')(conv4_3)

        ### Build the convolutional predictor layers on top of the base network

        # We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
        # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
        conv4_3_norm_mbox_conf = tf.layers.Conv2D(
            filters=n_boxes_conv4_3 * self.num_classes,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv4_3_norm_mbox_conf'
        )(conv4_3_norm)

        fc7_mbox_conf = tf.layers.Conv2D(
            filters=n_boxes_fc7 * self.num_classes,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='fc7_mbox_conf'
        )(fc7)

        conv6_2_mbox_conf = tf.layers.Conv2D(
            filters=n_boxes_conv6_2 * self.num_classes,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv6_2_mbox_conf'
        )(conv6_2)

        conv7_2_mbox_conf = tf.layers.Conv2D(
            filters=n_boxes_conv7_2 * self.num_classes,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv7_2_mbox_conf'
        )(conv7_2)

        conv8_2_mbox_conf = tf.layers.Conv2D(
            filters=n_boxes_conv8_2 * self.num_classes,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv8_2_mbox_conf'
        )(conv8_2)

        conv9_2_mbox_conf = tf.layers.Conv2D(
            filters=n_boxes_conv9_2 * self.num_classes,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv9_2_mbox_conf'
        )(conv9_2)

        # We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
        # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
        conv4_3_norm_mbox_loc = tf.layers.Conv2D(
            filters=n_boxes_conv4_3 * 4,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv4_3_norm_mbox_loc'
        )(conv4_3_norm)

        fc7_mbox_loc = tf.layers.Conv2D(
            filters=n_boxes_fc7 * 4,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='fc7_mbox_loc'
        )(fc7)

        conv6_2_mbox_loc = tf.layers.Conv2D(
            filters=n_boxes_conv6_2 * 4,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv6_2_mbox_loc'
        )(conv6_2)

        conv7_2_mbox_loc = tf.layers.Conv2D(
            filters=n_boxes_conv7_2 * 4,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv7_2_mbox_loc'
        )(conv7_2)

        conv8_2_mbox_loc = tf.layers.Conv2D(
            filters=n_boxes_conv8_2 * 4,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv8_2_mbox_loc'
        )(conv8_2)

        conv9_2_mbox_loc = tf.layers.Conv2D(
            filters=n_boxes_conv9_2 * 4,
            kernel_size=(3, 3),
            padding='same',
            kernel_initializer=tf.keras.initializers.he_normal(),
            name='conv9_2_mbox_loc'
        )(conv9_2)

        # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
        conv4_3_norm_mbox_priorbox = AnchorBoxes(
            self.image_height,
            self.image_width,
            this_scale=self.scales[0],
            next_scale=self.scales[1],
            aspect_ratios=aspect_ratios_conv4_3,
            two_boxes_for_ar1=self.two_boxes_for_ar1,
            variances=self.variances,
            coords=self.coords,
            normalize_coords=self.normalize_coords,
            name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)

        fc7_mbox_priorbox = AnchorBoxes(
            self.image_height,
            self.image_width,
            this_scale=self.scales[1],
            next_scale=self.scales[2],
            aspect_ratios=aspect_ratios_fc7,
            two_boxes_for_ar1=self.two_boxes_for_ar1,
            variances=self.variances,
            coords=self.coords,
            normalize_coords=self.normalize_coords,
            name='fc7_mbox_priorbox')(fc7_mbox_loc)

        conv6_2_mbox_priorbox = AnchorBoxes(
            self.image_height,
            self.image_width,
            this_scale=self.scales[2],
            next_scale=self.scales[3],
            aspect_ratios=aspect_ratios_conv6_2,
            two_boxes_for_ar1=self.two_boxes_for_ar1,
            variances=self.variances,
            coords=self.coords,
            normalize_coords=self.normalize_coords,
            name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)

        conv7_2_mbox_priorbox = AnchorBoxes(
            self.image_height, self.image_width,
            this_scale=self.scales[3],
            next_scale=self.scales[4],
            aspect_ratios=aspect_ratios_conv7_2,
            two_boxes_for_ar1=self.two_boxes_for_ar1,
            variances=self.variances,
            coords=self.coords,
            normalize_coords=self.normalize_coords,
            name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)

        conv8_2_mbox_priorbox = AnchorBoxes(
            self.image_height,
            self.image_width,
            this_scale=self.scales[4],
            next_scale=self.scales[5],
            aspect_ratios=aspect_ratios_conv8_2,
            two_boxes_for_ar1=self.two_boxes_for_ar1,
            variances=self.variances,
            coords=self.coords,
            normalize_coords=self.normalize_coords,
            name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)

        conv9_2_mbox_priorbox = AnchorBoxes(
            self.image_height,
            self.image_width,
            this_scale=self.scales[5],
            next_scale=self.scales[6],
            aspect_ratios=aspect_ratios_conv9_2,
            two_boxes_for_ar1=self.two_boxes_for_ar1,
            variances=self.variances,
            coords=self.coords,
            normalize_coords=self.normalize_coords,
            name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

        # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
        # We want the classes isolated in the last axis to perform softmax on them
        conv4_3_norm_mbox_conf_reshape = tf.reshape(
            conv4_3_norm_mbox_conf,
            (-1, self.num_classes),
            name="conv4_3_norm_mbox_conf_reshape")

        fc7_mbox_conf_reshape = tf.reshape(
            fc7_mbox_conf,
            (-1, self.num_classes),
            name='fc7_mbox_conf_reshape')

        conv6_2_mbox_conf_reshape = tf.reshape(
            conv6_2_mbox_conf,
            (-1, self.num_classes),
            name='conv6_2_mbox_conf_reshape')

        conv7_2_mbox_conf_reshape = tf.reshape(
            conv7_2_mbox_conf,
            (-1, self.num_classes),
            name='conv7_2_mbox_conf_reshape')

        conv8_2_mbox_conf_reshape = tf.reshape(
            conv8_2_mbox_conf,
            (-1, self.num_classes),
            name='conv8_2_mbox_conf_reshape')

        conv9_2_mbox_conf_reshape = tf.reshape(
            conv9_2_mbox_conf,
            (-1, self.num_classes),
            name='conv9_2_mbox_conf_reshape')

        # Reshape the box predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
        # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
        conv4_3_norm_mbox_loc_reshape = tf.reshape(
            conv4_3_norm_mbox_loc,
            (-1, 4),
            name='conv4_3_norm_mbox_loc_reshape')

        fc7_mbox_loc_reshape = tf.reshape(
            fc7_mbox_loc,
            (-1, 4),
            name='fc7_mbox_loc_reshape')

        conv6_2_mbox_loc_reshape = tf.reshape(
            conv6_2_mbox_loc,
            (-1, 4),
            name='conv6_2_mbox_loc_reshape')

        conv7_2_mbox_loc_reshape = tf.reshape(
            conv7_2_mbox_loc,
            (-1, 4),
            name='conv7_2_mbox_loc_reshape')

        conv8_2_mbox_loc_reshape = tf.reshape(
            conv8_2_mbox_loc,
            (-1, 4),
            name='conv8_2_mbox_loc_reshape')

        conv9_2_mbox_loc_reshape = tf.reshape(
            conv9_2_mbox_loc,
            (-1, 4),
            name='conv9_2_mbox_loc_reshape')

        # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
        conv4_3_norm_mbox_priorbox_reshape = tf.reshape(
            conv4_3_norm_mbox_priorbox,
            (-1, 8),
            name='conv4_3_norm_mbox_priorbox_reshape')

        fc7_mbox_priorbox_reshape = tf.reshape(
            fc7_mbox_priorbox,
            (-1, 8),
            name='fc7_mbox_priorbox_reshape')

        conv6_2_mbox_priorbox_reshape = tf.reshape(
            conv6_2_mbox_priorbox,
            (-1, 8),
            name='conv6_2_mbox_priorbox_reshape')

        conv7_2_mbox_priorbox_reshape = tf.reshape(
            conv7_2_mbox_priorbox,
            (-1, 8),
            name='conv7_2_mbox_priorbox_reshape')

        conv8_2_mbox_priorbox_reshape = tf.reshape(
            conv8_2_mbox_priorbox,
            (-1, 8),
            name='conv8_2_mbox_priorbox_reshape')

        conv9_2_mbox_priorbox_reshape = tf.reshape(
            conv9_2_mbox_priorbox,
            (-1, 8),
            name='conv9_2_mbox_priorbox_reshape')

        ### Concatenate the predictions from the different layers

        # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
        # so we want to concatenate along axis 1, the number of boxes per layer
        # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
        mbox_conf = tf.concat([
            conv4_3_norm_mbox_conf_reshape,
            fc7_mbox_conf_reshape,
            conv6_2_mbox_conf_reshape,
            conv7_2_mbox_conf_reshape,
            conv8_2_mbox_conf_reshape,
            conv9_2_mbox_conf_reshape],
            axis=0, name='mbox_conf')

        # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
        mbox_loc = tf.concat([
            conv4_3_norm_mbox_loc_reshape,
            fc7_mbox_loc_reshape,
            conv6_2_mbox_loc_reshape,
            conv7_2_mbox_loc_reshape,
            conv8_2_mbox_loc_reshape,
            conv9_2_mbox_loc_reshape],
            axis=0, name='mbox_loc')

        # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
        mbox_priorbox = tf.concat([
            conv4_3_norm_mbox_priorbox_reshape,
            fc7_mbox_priorbox_reshape,
            conv6_2_mbox_priorbox_reshape,
            conv7_2_mbox_priorbox_reshape,
            conv8_2_mbox_priorbox_reshape,
            conv9_2_mbox_priorbox_reshape],
            axis=0, name='mbox_priorbox')

        # The box coordinate predictions will go into the loss function just the way they are,
        # but for the class predictions, we'll apply a softmax activation layer first
        mbox_conf_softmax = tf.nn.softmax(mbox_conf, name='mbox_conf_softmax')

        # Concatenate the class and box predictions and the anchors to one large predictions vector
        # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
        predictions = tf.concat([
            mbox_conf_softmax,
            mbox_loc,
            mbox_priorbox],
            axis=1, name='predictions')
        predict_shape = predictions.get_shape().as_list()
        predictions = tf.reshape(
            predictions,
            shape=(self.batch_size,
                   predict_shape[0] // self.batch_size,
                   self.num_classes + 4 + 8))

        # Get the spatial dimensions (height, width) of the predictor conv layers, we need them to
        # be able to generate the default boxes for the matching process outside of the model during training.
        # Note that the original implementation performs anchor box matching inside the loss function. We don't do that.
        # Instead, we'll do it in the batch generator function.
        # The spatial dimensions are the same for the confidence and localization predictors, so we just take those of the conf layers.
        predictor_sizes = np.asarray([
            conv4_3_norm_mbox_conf.get_shape().as_list()[1:3],
            fc7_mbox_conf.get_shape().as_list()[1:3],
            conv6_2_mbox_conf.get_shape().as_list()[1:3],
            conv7_2_mbox_conf.get_shape().as_list()[1:3],
            conv8_2_mbox_conf.get_shape().as_list()[1:3],
            conv9_2_mbox_conf.get_shape().as_list()[1:3]])

        res = {
            "predictions": predictions,
            "predictor_sizes": predictor_sizes
        }
        return res

    def loss(self, y_true, y_pred):
        if self.model_loss_obj is None:
            self.model_loss_obj = Loss(
                neg_pos_ratio=self.neg_pos_ratio,
                n_neg_min=self.n_neg_min,
                alpha=self.loss_alpha)
        return self.model_loss_obj.compute_loss(y_true, y_pred)
