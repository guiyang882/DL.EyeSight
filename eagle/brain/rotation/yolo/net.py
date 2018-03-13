# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Net(object):
    def __init__(self, common_params, net_params):
        if not isinstance(common_params, dict):
            raise TypeError("common_params must be dict")
        if not isinstance(net_params, dict):
            raise TypeError("net_params must be dict")

    def inference(self, images):
        """Build the yolo model
        Args:
          images:  4-D tensor [batch_size, image_height, image_width, channels]
        Returns:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, num_classes + 5 * boxes_per_cell]
        """
        raise NotImplementedError

    def loss(self, predicts, labels, objects_num):
        """Add Loss to all the trainable variables
        Args:
          predicts: 4-D tensor [batch_size, cell_size, cell_size, 5 * boxes_per_cell]
          ===> (num_classes, boxes_per_cell, 4 * boxes_per_cell)
          labels  : 3-D tensor of [batch_size, max_objects, 5]
          objects_num: 1-D tensor [batch_size]
        """
        raise NotImplementedError
