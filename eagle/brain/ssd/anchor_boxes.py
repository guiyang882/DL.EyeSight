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

import keras.backend as K
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec

from eagle.brain.ssd.box_encode_decode_utils import convert_coordinates


class AnchorBoxes(Layer):
    '''
    Input shape:
        4D tensor of shape 
            `(batch, channels, height, width)` if dim_ordering == 'th'
            `(batch, height, width, channels)` if dim_ordering == 'tf'
    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 8)`.
        The last axis contains the four anchor box coordinates and the four variance values for each box.
    '''
    def __init__(self,
                 img_height, img_width,
                 this_scale, next_scale,
                 aspect_ratios=[0.5, 1.0, 2.0],
                 two_boxes_for_ar1=True,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 coords='centroids', normalize_coords=False, **kwargs):
        '''
        this_scale (float): A float in [0, 1], the scaling factor for the size of the generated anchor boxes
                as a fraction of the shorter side of the input image.
        next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
        aspect_ratios (list, optional): The list of aspect ratios for which default boxes are to be
                generated for this layer. Defaults to [0.5, 1.0, 2.0].
        two_boxes_for_ar1 (bool, optional): Only relevant if `aspect_ratios` contains 1.
                If `True`, two default boxes will be generated for aspect ratio 1. The first will be generated
                using the scaling factor for the respective layer, the second one will be generated using
                geometric mean of said scaling factor and next bigger scaling factor. Defaults to `True`.
        variances (list, optional): A list of 4 floats >0 with scaling factors (actually it's not factors but divisors
                to be precise) for the encoded predicted box coordinates. A variance value of 1.0 would apply
                no scaling at all to the predictions, while values in (0,1) upscale the encoded predictions and values greater
                than 1.0 downscale the encoded predictions. If you want to reproduce the configuration of the original SSD,
                set this to `[0.1, 0.1, 0.2, 0.2]`, provided the coordinate Others is 'centroids'. Defaults to `[1.0, 1.0, 1.0, 1.0]`.
        coords (str, optional): The box coordinate Others to be used. Can be either 'centroids' for the Others
                `(cx, cy, w, h)` (box center coordinates, width, and height) or 'minmax' for the Others
                `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.
        normalize_coords (bool, optional): Set to `True` if the model uses relative instead of absolute coordinates,
                i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates. Defaults to `False`.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError("Tensorflow Model, Not %s".format(K.backend()))
        if (this_scale<0) or (this_scale>1) or (next_scale<0):
            raise ValueError("this_scale or next_scale must be in [0, 1]")

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.next_scale = next_scale
        self.aspect_ratios = aspect_ratios
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.variances = variances
        self.coords = coords
        self.normalize_coords = normalize_coords
        
        # Compute the number of boxes per cell
        if (1 in aspect_ratios) & two_boxes_for_ar1:
            self.n_boxes = len(aspect_ratios) + 1
        else:
            self.n_boxes = len(aspect_ratios)

        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        self.aspect_ratios = np.sort(self.aspect_ratios)
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        for ar in self.aspect_ratios:
            if (ar == 1) & self.two_boxes_for_ar1:
                # Compute the regular default box for aspect ratio 1 and...
                w = self.this_scale * size * np.sqrt(ar)
                h = self.this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))
                # ...also compute one slightly larger version using the geometric mean of this scale value and the next
                w = np.sqrt(self.this_scale * self.next_scale) * size * np.sqrt(ar)
                h = np.sqrt(self.this_scale * self.next_scale) * size / np.sqrt(ar)
                wh_list.append((w, h))
            else:
                w = self.this_scale * size * np.sqrt(ar)
                h = self.this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))
        wh_list = np.array(wh_list)

        # We need the shape of the input tensor
        batch_size, feature_map_height, feature_map_width, feature_map_channels = x.get_shape().as_list()

        # Compute the grid of box center points. They are identical for all aspect ratios
        cell_width = self.img_width / feature_map_width
        cell_height = self.img_height / feature_map_height
        cx = np.linspace(cell_width/2, self.img_width-cell_width/2, feature_map_width)
        cy = np.linspace(cell_height/2, self.img_height-cell_height/2, feature_map_height)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        # This is necessary for np.tile() to do what we want further down
        cx_grid = np.expand_dims(cx_grid, -1)
        # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(
            boxes_tensor, start_index=0, conversion='centroids2minmax')

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, :2] /= self.img_width
            boxes_tensor[:, :, :, 2:] /= self.img_height

        if self.coords == 'centroids':
            # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth
            # Convert `(xmin, xmax, ymin, ymax)` back to `(cx, cy, w, h)`
            boxes_tensor = convert_coordinates(
                boxes_tensor, start_index=0, conversion='minmax2centroids')

        # 4: Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
        #    as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
        # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        variances_tensor = np.zeros_like(boxes_tensor)
        # Long live broadcasting
        variances_tensor += self.variances
        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = tf.tile(
            tf.constant(boxes_tensor, dtype='float32'),
            (x.get_shape().as_list()[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape
        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 8)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'next_scale': self.next_scale,
            'aspect_ratios': list(self.aspect_ratios),
            'two_boxes_for_ar1': self.two_boxes_for_ar1,
            'variances': list(self.variances),
            'coords': self.coords,
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
