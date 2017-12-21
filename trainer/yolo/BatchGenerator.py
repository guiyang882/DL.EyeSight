# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/20

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import PIL
import numpy as np
import sklearn

from .feature_base_yolo import preprocess_true_boxes


def process_data(images, boxes=None):
    '''processes the data'''
    images = [PIL.Image.fromarray(i) for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image / 127.5 - 1 for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

class BatchGenerator:
    def __init__(self, image_data, boxes_data, anchors):
        self.image_data = image_data
        self.boxes_data = boxes_data

        assert (len(image_data) == len(boxes_data))
        self.anchors = anchors
        self.idx = [i for i in range(0, len(image_data))]

    def get_n_samples(self):
        return len(self.image_data)

    def generate(self, batch_size=32, shuffle=True, train=True):
        if shuffle:
            self.idx = sklearn.utils.shuffle(self.idx)

        cur_idx = 0
        while True:
            batch_imgs, batch_boxes, batch_masks, batch_true_boxes = [], [], [], []

            if cur_idx >= len(self.image_data):
                cur_idx = 0
                if shuffle:
                    self.idx = sklearn.utils.shuffle(self.idx)

            if cur_idx + batch_size >= len(self.image_data):
                cur_idx = len(self.image_data) - batch_size
            src_imgs = [self.image_data[idx] for idx in self.idx[cur_idx:cur_idx+batch_size]]
            src_boxes = [self.boxes_data[idx] for idx in self.idx[cur_idx:cur_idx+batch_size]]
            batch_imgs, batch_boxes = process_data(src_imgs, src_boxes)
            batch_masks, batch_true_boxes = get_detector_mask(batch_boxes, self.anchors)
            if train:
                yield ([np.array(batch_imgs), np.array(batch_boxes), np.array(batch_masks), np.array(batch_true_boxes)], np.zeros(len(batch_size)))
            cur_idx += batch_size
