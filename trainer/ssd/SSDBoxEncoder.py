# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/18

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

'''
A class to transform ground truth labels for object detection in images
(2D bounding box coordinates and class labels) to the format required for
training an SSD model, and to transform predictions of the SSD model back
to the original format of the input labels.
In the process of encoding ground truth labels, a template of anchor boxes
is being built, which are subsequently matched to the ground truth boxes
via an intersection-over-union threshold criterion.
'''

import numpy as np
from .box_encode_decode_utils import iou, convert_coordinates


class SSDBoxEncoder:

    def __init__(self, img_height, img_width, n_classes,
                 predictor_sizes,
                 min_scale=0.1, max_scale=0.9, scales=None,
                 aspect_ratios_global=[0.5, 1.0, 2.0], aspect_ratios_per_layer=None,
                 two_boxes_for_ar1=True, limit_boxes=True,
                 variances=[1.0, 1.0, 1.0, 1.0],
                 pos_iou_threshold=0.5, neg_iou_threshold=0.3,
                 coords='centroids', normalize_coords=False):
        '''
        two_boxes_for_ar1 (bool, optional): Only relevant for aspect ratios lists that contain 1. Will be ignored otherwise.
            If `True`, two anchor boxes will be generated for aspect ratio 1. The first will be generated
            using the scaling factor for the respective layer, the second one will be generated using
            geometric mean of said scaling factor and next bigger scaling factor.
        limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries.
            Defaults to `True`.
        variances (list, optional): A list of 4 floats >0 with scaling factors (actually it's not factors but divisors
            to be precise) for the encoded ground truth (i.e. target) box coordinates. A variance value of 1.0 would apply
            no scaling at all to the targets, while values in (0,1) upscale the encoded targets and values greater than 1.0
            downscale the encoded targets. If you want to reproduce the configuration of the original SSD,
            set this to `[0.1, 0.1, 0.2, 0.2]`, provided the coordinate format is 'centroids'. Defaults to `[1.0, 1.0, 1.0, 1.0]`.
        pos_iou_threshold (float, optional): The intersection-over-union similarity threshold that must be
            met in order to match a given ground truth box to a given anchor box. Defaults to 0.5.
        neg_iou_threshold (float, optional): The maximum allowed intersection-over-union similarity of an
            anchor box with any ground truth box to be labeled a negative (i.e. background) box. If an
            anchor box is neither a positive, nor a negative box, it will be ignored during training.
        coords (str, optional): The box coordinate format to be used internally in the model (i.e. this is not the input format
            of the ground truth labels). Can be either 'centroids' for the format `(cx, cy, w, h)` (box center coordinates, width,
            and height) or 'minmax' for the format `(xmin, xmax, ymin, ymax)`. Defaults to 'centroids'.
        normalize_coords (bool, optional): If `True`, the encoder uses relative instead of absolute coordinates.
            This means instead of using absolute tartget coordinates, the encoder will scale all coordinates to be within [0,1].
            This way learning becomes independent of the input image size. Defaults to `False`.
        '''
        predictor_sizes = np.array(predictor_sizes)
        if len(predictor_sizes.shape) == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError(
                "Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            # Must be two nested `if` statements since `list` and `bool` cannot
            # be combined by `&`
            if (len(scales) != len(predictor_sizes)+1):
                raise ValueError("It must be either scales is None or len(scales) == len(predictor_sizes)+1, but len(scales) == {} and len(predictor_sizes)+1 == {}".format(
                    len(scales), len(predictor_sizes)+1))
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError(
                    "All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
        else:  # If no list of scales was passed, we need to make sure that `min_scale` and `max_scale` are valid values.
            if not 0 < min_scale <= max_scale:
                raise ValueError("It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(
                    min_scale, max_scale))

        if aspect_ratios_per_layer:
            # Must be two nested `if` statements since `list` and `bool` cannot
            # be combined by `&`
            if (len(aspect_ratios_per_layer) != len(predictor_sizes)):
                raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == len(predictor_sizes), but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(
                    len(aspect_ratios_per_layer), len(predictor_sizes)))
            for aspect_ratios in aspect_ratios_per_layer:
                aspect_ratios = np.array(aspect_ratios)
                if np.any(aspect_ratios <= 0):
                    raise ValueError(
                        "All aspect ratios must be greater than zero.")
        else:
            if not aspect_ratios_global:
                raise ValueError(
                    "At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` cannot be `None`.")
            aspect_ratios_global = np.array(aspect_ratios_global)
            if np.any(aspect_ratios_global <= 0):
                raise ValueError(
                    "All aspect ratios must be greater than zero.")

        if len(variances) != 4:
            raise ValueError(
                "4 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError(
                "All variances must be >0, but the variances given are {}".format(variances))

        if neg_iou_threshold > pos_iou_threshold:
            raise ValueError(
                "It cannot be `neg_iou_threshold > pos_iou_threshold`.")

        if not (coords == 'minmax' or coords == 'centroids'):
            raise ValueError(
                "Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales = scales
        self.aspect_ratios_global = aspect_ratios_global
        self.aspect_ratios_per_layer = aspect_ratios_per_layer
        self.two_boxes_for_ar1 = two_boxes_for_ar1
        self.limit_boxes = limit_boxes
        self.variances = variances
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.coords = coords
        self.normalize_coords = normalize_coords

        # Compute the number of boxes per cell
        if aspect_ratios_per_layer:
            self.n_boxes = []
            for aspect_ratios in aspect_ratios_per_layer:
                if (1 in aspect_ratios) & two_boxes_for_ar1:
                    self.n_boxes.append(len(aspect_ratios) + 1)
                else:
                    self.n_boxes.append(len(aspect_ratios))
        else:
            if (1 in aspect_ratios_global) & two_boxes_for_ar1:
                self.n_boxes = len(aspect_ratios_global) + 1
            else:
                self.n_boxes = len(aspect_ratios_global)

    def generate_anchor_boxes(self, batch_size, feature_map_size,
                              aspect_ratios, this_scale, next_scale, diagnostics=False):
        """
        Arguments:
                batch_size (int): The batch size.
            feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
            next_scale (float): A float in [0, 1], the next larger scaling factor. Only relevant if
                `self.two_boxes_for_ar1 == True`.
            diagnostics (bool, optional): If true, two additional outputs will be returned.
                1) An array containing `(width, height)` for each box aspect ratio.
                2) A tuple `(cell_height, cell_width)` meaning how far apart the box centroids are placed
                   vertically and horizontally.
                This information is useful to understand in just a few numbers what the generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense
                their distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects
                to be detected.
        Returns:
            A 4D Numpy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 4)` where the
            last dimension contains `(xmin, xmax, ymin, ymax)` for each anchor box in each cell of the feature map.
        """
        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h`
        # using `scale` and `aspect_ratios`.
        aspect_ratios = np.sort(aspect_ratios)
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wh_list = []
        n_boxes = len(aspect_ratios)
        for ar in aspect_ratios:
            if (ar == 1) & self.two_boxes_for_ar1:
                # Compute the regular anchor box for aspect ratio 1 and...
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))
                # ...also compute one slightly larger version using the geometric mean of this scale value and the next
                w = np.sqrt(this_scale * next_scale) * size * np.sqrt(ar)
                h = np.sqrt(this_scale * next_scale) * size / np.sqrt(ar)
                wh_list.append((w, h))
                # Add 1 to `n_boxes` since we seem to have two boxes for aspect
                # ratio 1
                n_boxes += 1
            else:
                w = this_scale * size * np.sqrt(ar)
                h = this_scale * size / np.sqrt(ar)
                wh_list.append((w, h))
        wh_list = np.array(wh_list)

        # Compute the grid of box center points. They are identical for all
        # aspect ratios
        cell_height = self.img_height / feature_map_size[0]
        cell_width = self.img_width / feature_map_size[1]
        cx = np.linspace(cell_width/2, self.img_width -
                         cell_width/2, feature_map_size[1])
        cy = np.linspace(cell_height/2, self.img_height -
                         cell_height/2, feature_map_size[0])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        # This is necessary for np.tile() to do what we want further down
        cx_grid = np.expand_dims(cx_grid, -1)
        # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1)

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
        # where the last dimension will contain `(cx, cy, w, h)`
        boxes_tensor = np.zeros(
            (feature_map_size[0], feature_map_size[1], n_boxes, 4))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
        boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
        boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

        # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
        boxes_tensor = convert_coordinates(
            boxes_tensor, start_index=0, conversion='centroids2minmax')

        # If `limit_boxes` is enabled, clip the coordinates to lie within the
        # image boundaries
        if self.limit_boxes:
            x_coords = boxes_tensor[:, :, :, [0, 1]]
            x_coords[x_coords >= self.img_width] = self.img_width - 1
            x_coords[x_coords < 0] = 0
            boxes_tensor[:, :, :, [0, 1]] = x_coords
            y_coords = boxes_tensor[:, :, :, [2, 3]]
            y_coords[y_coords >= self.img_height] = self.img_height - 1
            y_coords[y_coords < 0] = 0
            boxes_tensor[:, :, :, [2, 3]] = y_coords

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, :2] /= self.img_width
            boxes_tensor[:, :, :, 2:] /= self.img_height

        if self.coords == 'centroids':
            # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth
            # Convert `(xmin, xmax, ymin, ymax)` back to `(cx, cy, w, h)`
            boxes_tensor = convert_coordinates(
                boxes_tensor, start_index=0, conversion='minmax2centroids')

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size,
        # feature_map_height, feature_map_width, n_boxes, 4)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = np.tile(boxes_tensor, (batch_size, 1, 1, 1, 1))

        # Now reshape the 5D tensor above into a 3D tensor of shape
        # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
        # order of the tensor content will be identical to the order obtained from the reshaping operation
        # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
        # use the same default index order, which is C-like index ordering)
        boxes_tensor = np.reshape(boxes_tensor, (batch_size, -1, 4))

        if diagnostics:
            return boxes_tensor, wh_list, (int(cell_height), int(cell_width))
        else:
            return boxes_tensor

    def generate_encode_template(self, batch_size, diagnostics=False):
        '''
        Produces an encoding template for the ground truth label tensor for a given batch.
        Note that all tensor creation, reshaping and concatenation operations performed in this function
        and the sub-functions it calls are identical to those performed inside the conv net model. This, of course,
        must be the case in order to preserve the spatial meaning of each box prediction, but it's useful to make
        yourself aware of this fact and why it is necessary.
        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that `y_encoded`
        has this specific form.
        Arguments:
            batch_size (int): The batch size.
            diagnostics (bool, optional): See the documnentation for `generate_anchor_boxes()`. The diagnostic output
                here is similar, just for all predictor conv layers.
        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 12)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 12` because the model
            output contains not only the 4 predicted box coordinate offsets, but also the 4 coordinates for
            the anchor boxes and the 4 variance values.
        '''

        # 1: Get the anchor box scaling factors for each conv layer from which we're going to make predictions
        # If `scales` is given explicitly, we'll use that instead of computing
        # it from `min_scale` and `max_scale`
        if self.scales is None:
            self.scales = np.linspace(
                self.min_scale, self.max_scale, len(self.predictor_sizes)+1)

        # 2: For each conv predictor layer (i.e. for each scale factor) get the tensors for
        #    the anchor box coordinates of shape `(batch, n_boxes_total, 4)`
        boxes_tensor = []
        if diagnostics:
            wh_list = []  # List to hold the box widths and heights
            cell_sizes = []  # List to hold horizontal and vertical distances between any two boxes
            if self.aspect_ratios_per_layer:  # If individual aspect ratios are given per layer, we need to pass them to `generate_anchor_boxes()` accordingly
                for i in range(len(self.predictor_sizes)):
                    boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                  feature_map_size=self.predictor_sizes[
                                                                      i],
                                                                  aspect_ratios=self.aspect_ratios_per_layer[
                                                                      i],
                                                                  this_scale=self.scales[
                                                                      i],
                                                                  next_scale=self.scales[
                                                                      i+1],
                                                                  diagnostics=True)
                    boxes_tensor.append(boxes)
                    wh_list.append(wh)
                    cell_sizes.append(cells)
            else:  # Use the same global aspect ratio list for all layers
                for i in range(len(self.predictor_sizes)):
                    boxes, wh, cells = self.generate_anchor_boxes(batch_size=batch_size,
                                                                  feature_map_size=self.predictor_sizes[
                                                                      i],
                                                                  aspect_ratios=self.aspect_ratios_global,
                                                                  this_scale=self.scales[
                                                                      i],
                                                                  next_scale=self.scales[
                                                                      i+1],
                                                                  diagnostics=True)
                    boxes_tensor.append(boxes)
                    wh_list.append(wh)
                    cell_sizes.append(cells)
        else:
            if self.aspect_ratios_per_layer:
                for i in range(len(self.predictor_sizes)):
                    boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                                   feature_map_size=self.predictor_sizes[
                                                                       i],
                                                                   aspect_ratios=self.aspect_ratios_per_layer[
                                                                       i],
                                                                   this_scale=self.scales[
                                                                       i],
                                                                   next_scale=self.scales[
                                                                       i+1],
                                                                   diagnostics=False))
            else:
                for i in range(len(self.predictor_sizes)):
                    boxes_tensor.append(self.generate_anchor_boxes(batch_size=batch_size,
                                                                   feature_map_size=self.predictor_sizes[
                                                                       i],
                                                                   aspect_ratios=self.aspect_ratios_global,
                                                                   this_scale=self.scales[
                                                                       i],
                                                                   next_scale=self.scales[
                                                                       i+1],
                                                                   diagnostics=False))

        # Concatenate the anchor tensors from the individual layers to one
        boxes_tensor = np.concatenate(boxes_tensor, axis=1)

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        # It will contain all zeros for now, the classes will be set in the
        # matching process that follows
        classes_tensor = np.zeros(
            (batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        # contains the same 4 variance values for every position in the last
        # axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances  # Long live broadcasting

        # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encode_template` has the same
        #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        #    `boxes_tensor` a second time.
        y_encode_template = np.concatenate(
            (classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encode_template, wh_list, cell_sizes
        else:
            return y_encode_template

    def encode_y(self, ground_truth_labels, diagnostics=False):
        '''
        Convert ground truth bounding box data into a suitable format to train an SSD model.
        For each image in the batch, each ground truth bounding box belonging to that image will be compared against each
        anchor box in a template with respect to their jaccard similarity. If the jaccard similarity is greater than
        or equal to the set threshold, the boxes will be matched, meaning that the ground truth box coordinates and class
        will be written to the the specific position of the matched anchor box in the template.
        The class for all anchor boxes for which there was no match with any ground truth box will be set to the
        background class, except for those anchor boxes whose IoU similarity with any ground truth box is higher than
        the set negative threshold (see the `neg_iou_threshold` argument in `__init__()`).
        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, xmin, xmax, ymin, ymax)`, and `class_id` must be an integer greater than 0 for all boxes
                as class_id 0 is reserved for the background class.
            diagnostics (bool, optional): If `True`, not only the encoded ground truth tensor will be returned,
                but also a copy of it with anchor box coordinates in place of the ground truth coordinates.
                This can be very useful if you want to visualize which anchor boxes got matched to which ground truth
                boxes.
        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 4 + 4 + 4)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The four elements after the class vecotrs in
            the last axis are the box coordinates, the next four elements after that are just dummy elements, and
            the last four elements are the variances.
        '''

        # 1: Generate the template for y_encoded
        y_encode_template = self.generate_encode_template(
            batch_size=len(ground_truth_labels), diagnostics=False)
        # We'll write the ground truth box data to this array
        y_encoded = np.copy(y_encode_template)

        # 2: Match the boxes from `ground_truth_labels` to the anchor boxes in `y_encode_template`
        #    and for each matched box record the ground truth coordinates in `y_encoded`.
        # Every time there is no match for a anchor box, record `class_id` 0 in
        # `y_encoded` for that anchor box.

        # An identity matrix that we'll use as one-hot class vectors
        class_vector = np.eye(self.n_classes)

        for i in range(y_encode_template.shape[0]):  # For each batch item...
            # 1 for all anchor boxes that are not yet matched to a ground truth
            # box, 0 otherwise
            available_boxes = np.ones((y_encode_template.shape[1]))
            # 1 for all negative boxes, 0 otherwise
            negative_boxes = np.ones((y_encode_template.shape[1]))
            # For each ground truth box belonging to the current batch item...
            for true_box in ground_truth_labels[i]:
                true_box = true_box.astype(np.float)
                if abs(true_box[2] - true_box[1] < 0.001) or abs(true_box[4] - true_box[3] < 0.001):
                    continue  # Protect ourselves against bad ground truth data: boxes with width or height equal to zero
                if self.normalize_coords:
                    # Normalize xmin and xmax to be within [0,1]
                    true_box[1:3] /= self.img_width
                    # Normalize ymin and ymax to be within [0,1]
                    true_box[3:5] /= self.img_height
                if self.coords == 'centroids':
                    true_box = convert_coordinates(
                        true_box, start_index=1, conversion='minmax2centroids')
                # The iou similarities for all anchor boxes
                similarities = iou(y_encode_template[
                                   i, :, -12:-8], true_box[1:], coords=self.coords)
                # If a negative box gets an IoU match >=
                # `self.neg_iou_threshold`, it's no longer a valid negative box
                negative_boxes[similarities >= self.neg_iou_threshold] = 0
                # Filter out anchor boxes which aren't available anymore (i.e.
                # already matched to a different ground truth box)
                similarities *= available_boxes
                available_and_thresh_met = np.copy(similarities)
                # Filter out anchor boxes which don't meet the iou threshold
                available_and_thresh_met[
                    available_and_thresh_met < self.pos_iou_threshold] = 0
                # Get the indices of the left-over anchor boxes to which we
                # want to assign this ground truth box
                assign_indices = np.nonzero(available_and_thresh_met)[0]
                if len(assign_indices) > 0:  # If we have any matches
                    # Write the ground truth box coordinates and class to all
                    # assigned anchor box positions. Remember that the last
                    # four elements of `y_encoded` are just dummy entries.
                    y_encoded[i, assign_indices, :-8] = np.concatenate(
                        (class_vector[int(true_box[0])], true_box[1:]), axis=0)
                    # Make the assigned anchor boxes unavailable for the next
                    # ground truth box
                    available_boxes[assign_indices] = 0
                else:  # If we don't have any matches
                    # Get the index of the best iou match out of all available
                    # boxes
                    best_match_index = np.argmax(similarities)
                    # Write the ground truth box coordinates and class to the
                    # best match anchor box position
                    y_encoded[i, best_match_index, :-8] = np.concatenate(
                        (class_vector[int(true_box[0])], true_box[1:]), axis=0)
                    # Make the assigned anchor box unavailable for the next
                    # ground truth box
                    available_boxes[best_match_index] = 0
                    # The assigned anchor box is no longer a negative box
                    negative_boxes[best_match_index] = 0
            # Set the classes of all remaining available anchor boxes to class
            # zero
            background_class_indices = np.nonzero(negative_boxes)[0]
            y_encoded[i, background_class_indices, 0] = 1

        # 3: Convert absolute box coordinates to offsets from the anchor boxes
        # and normalize them
        if self.coords == 'centroids':
            # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
            y_encoded[:, :, [-12, -11]] -= y_encode_template[:, :, [-12, -11]]
            # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
            y_encoded[:, :, [-12, -11]] /= y_encode_template[:,
                                                             :, [-10, -9]] * y_encode_template[:, :, [-4, -3]]
            y_encoded[:, :, [-10, -9]] /= y_encode_template[:, :,
                                                            [-10, -9]]  # w(gt) / w(anchor), h(gt) / h(anchor)
            # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) /
            # h_variance (ln == natural logarithm)
            y_encoded[
                :, :, [-10, -9]] = np.log(y_encoded[:, :, [-10, -9]]) / y_encode_template[:, :, [-2, -1]]
        else:
            # (gt - anchor) for all four coordinates
            y_encoded[:, :, -12:-8] -= y_encode_template[:, :, -12:-8]
            # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
            y_encoded[:, :, [-12, -11]] /= np.expand_dims(
                y_encode_template[:, :, -11] - y_encode_template[:, :, -12], axis=-1)
            # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
            y_encoded[:, :, [-10, -9]] /= np.expand_dims(
                y_encode_template[:, :, -9] - y_encode_template[:, :, -10], axis=-1)
            # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
            y_encoded[:, :, -12:-8] /= y_encode_template[:, :, -4:]

        if diagnostics:
            # Here we'll save the matched anchor boxes (i.e. anchor boxes that
            # were matched to a ground truth box, but keeping the anchor box
            # coordinates).
            y_matched_anchors = np.copy(y_encoded)
            # Keeping the anchor box coordinates means setting the offsets to
            # zero.
            y_matched_anchors[:, :, -12:-8] = 0
            return y_encoded, y_matched_anchors
        else:
            return y_encoded
