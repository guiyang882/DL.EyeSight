# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/26

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
from enum import Enum

import eagle.utils as eu


class BatchStatus(Enum):
    IMG_AUG_BATCH = "image.aug.Batch"
    NP_ARRAY = "numpy.array"
    EMPTY_LIST = "empty.list"
    NP_ARRAYS = "numpy.array.list"
    KPS_ON_IMAGE = "image.aug.KeyPointsOnImage"



class KeyPoint(object):
    """
    A single keypoint (aka landmark) on an image.

    Parameters
    ----------
    x : number
        Coordinate of the keypoint on the x axis.

    y : number
        Coordinate of the keypoint on the y axis.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def x_int(self):
        return int(round(self.x))

    @property
    def y_int(self):
        return int(round(self.y))

    def project(self, from_shape, to_shape):
        """
        Project the KeyPoint onto a new position on a new image.

        E.g. if the keypoint is on its original image at x=(10 of 100 pixels)
        and y=(20 of 100 pixels) and is projected onto a new image with
        size (width=200, height=200), its new position will be (20, 40).

        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).

        Parameters
        ----------
        from_shape : tuple
            Shape of the original image. (Before resize.)

        to_shape : tuple
            Shape of the new image. (After resize.)

        Returns
        -------
        out : KeyPoint
            KeyPoint object with new coordinates.
        """
        if from_shape[:2] == to_shape[:2]:
            return KeyPoint(x=self.x, y=self.y)
        else:
            from_height, from_width = from_shape[:2]
            to_height, to_width = to_shape[:2]
            x = (self.x / from_width) * to_width
            y = (self.y / from_height) * to_height
            return KeyPoint(x=x, y=y)

    def shift(self, x, y):
        """
        Move the KeyPoint around on an image.

        Parameters
        ----------
        x : number
            Move by this value on the x axis.

        y : number
            Move by this value on the y axis.

        Returns
        -------
        out : KeyPoint
            KeyPoint object with new coordinates.
        """
        return KeyPoint(self.x + x, self.y + y)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "KeyPoint(x=%.5f, y=%.5f)" % (self.x, self.y)


class KeyPointsOnImage(object):
    """
    Object that represents all keypoints on a single image.

    Parameters
    ----------
    keypoints : list of Keypoint
        List of keypoints on the image.

    shape : tuple of int
        The shape of the image on which the keypoints are placed.

    Examples
    --------
    >>> kps = [KeyPoint(x=10, y=20), KeyPoint(x=34, y=60)]
    >>> kps_oi = KeyPointsOnImage(kps, shape=image.shape)
    """
    def __init__(self, keypoints, shape):
        self.keypoints = keypoints
        eu.do_assert(isinstance(shape, (tuple, list)))
        self.shape = tuple(shape)

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    def project(self, image):
        """
        Project keypoints from one image to a new one.

        Parameters
        ----------
        image : ndarray or tuple
            New image onto which the keypoints are to be projected.
            May also simply be that new image's shape tuple.

        Returns
        -------
        keypoints : KeypointsOnImage
            Object containing all projected keypoints.
        """
        if eu.is_np_array(image):
            shape = image.shape
        else:
            shape = image

        if shape[:2] == self.shape[:2]:
            return self.deepcopy()
        else:
            keypoints = [kp.project(self.shape, shape) for kp in self.keypoints]
            return KeyPointsOnImage(keypoints, shape)

    def shift(self, x, y):
        """
        Move the keypoints around on an image.

        Parameters
        ----------
        x : number
            Move each keypoint by this value on the x axis.

        y : number
            Move each keypoint by this value on the y axis.

        Returns
        -------
        out : KeypointsOnImage
            Keypoints after moving them.
        """
        kps = [kp.shift(x=x, y=y) for kp in self.keypoints]
        return KeyPointsOnImage(kps, self.shape)

    def get_coords_array(self):
        """
        Convert the coordinates of all keypoints in this object to
        an array of shape (N,2).

        Returns
        -------
        result : (N, 2) ndarray
            Where N is the number of keypoints. Each first value is the
            x coordinate, each second value is the y coordinate.
        """
        results = np.zeros((len(self.keypoints), 2), np.float32)
        for i, kp in enumerate(self.keypoints):
            results[i, 0] = kp.x
            results[i, 1] = kp.y
        return results

    @staticmethod
    def from_coords_array(coords, shape):
        """
        Convert an array (N,2) with a given image shape to a KeypointsOnImage
        object.

        Parameters
        ----------
        coords : (N, 2) ndarray
            Coordinates of N keypoints on the original image.
            Each first entry (i, 0) is expected to be the x coordinate.
            Each second entry (i, 1) is expected to be the y coordinate.

        shape : tuple
            Shape tuple of the image on which the keypoints are placed.

        Returns
        -------
        out : KeypointsOnImage
            KeypointsOnImage object that contains all keypoints from the array.
        """
        kps = [KeyPoint(x=coords[i, 0], y=coords[i, 1])
               for i in range(coords.shape[0])]
        return KeyPointsOnImage(kps, shape)

    def draw_on_image(self, image, color=(0, 255, 0), size=3, copy=True):
        """
        Draw all keypoints onto a given image. Each keypoint is marked by a
        square of a chosen color and size.

        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the keypoints.
            This image should usually have the same shape as
            set in KeypointsOnImage.shape.

        color : int or list of ints or tuple of ints or (3,) ndarray, optional(default=[0, 255, 0])
            The RGB color of all keypoints. If a single int `C`, then that is
            equivalent to (C,C,C).

        size : int, optional(default=3)
            The size of each point. If set to C, each square will have
            size CxC.

        copy : bool, optional(default=True)
            Whether to copy the image before drawing the points.

        Returns
        -------
        image : (H,W,3) ndarray
            Image with drawn keypoints.
        """
        if copy:
            image = np.copy(image)
        height, width = image.shape[:2]
        for kp in self.keypoints:
            y, x = kp.y_int, kp.x_int
            if 0 <= y < height and 0 <= x < width:
                x1 = max(x - size // 2, 0)
                x2 = min(x + 1 + size // 2, width - 1)
                y1 = max(y - size // 2, 0)
                y2 = min(y + 1 + size // 2, height - 1)
                image[y1:y2, x1:x2] = list(color)
        return image

    def to_keypoint_image(self, size=1):
        """
        Draws a new black image of shape (H,W,N) in which all keypoint coordinates
        are set to 255.
        (H=shape height, W=shape width, N=number of keypoints)

        This function can be used as a helper when augmenting keypoints with
        a method that only supports the augmentation of images.

        Parameters
        -------
        size : int
            Size of each (squared) point.

        Returns
        -------
        image : (H,W,N) ndarray
            Image in which the keypoints are marked. H is the height,
            defined in KeypointsOnImage.shape[0] (analogous W). N is the
            number of keypoints.
        """
        eu.do_assert(len(self.keypoints) > 0)
        height, width = self.shape[:2]
        image = np.zeros((height, width, len(self.keypoints)), dtype=np.uint8)
        sizeh = max(0, (size - 1) // 2)
        for i, kp in enumerate(self.keypoints):
            y, x = kp.y_int, kp.x_int
            x1 = np.clip(x - sizeh, 0, width - 1)
            x2 = np.clip(x + sizeh + 1, 0, width - 1)
            y1 = np.clip(y - sizeh, 0, height - 1)
            y2 = np.clip(y + sizeh + 1, 0, height - 1)

            if x1 < x2 and y1 < y2:
                image[y1:y2, x1:x2] = 128
            if 0 <= y < height and 0 <= x < width:
                image[y, x, i] = 255
        return image

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        kps = [KeyPoint(x=kp.x, y=kp.y) for kp in self.keypoints]
        return KeyPointsOnImage(kps, tuple(self.shape))

    def __repr(self):
        return self.__str__()

    def __str(self):
        return "KeyPointsOnImage(%s, shape=%s)" % (
            str(self.keypoints), self.shape)


class BoundingBox(object):
    def __init__(self, x1, x2, y1, y2):
        if x1 > x2:
            x2, x1 = x1, x2
        if y1 > y2:
            y2, y1 = y1, y2
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    @property
    def left_up_pos(self):
        return (int(round(self.x1)), int(round(self.y1)))

    @property
    def right_down_pos(self):
        return (int(round(self.x2)), int(round(self.y2)))

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def center_x(self):
        return self.x1 + self.width / 2

    @property
    def center_y(self):
        return self.y1 + self.height / 2

    @property
    def area(self):
        return self.width * self.height

    def project(self, from_shape, to_shape):
        """
        Project the bounding box onto a new position on a new image.

        E.g. if the bounding box is on its original image at
        x1=(10 of 100 pixels) and y1=(20 of 100 pixels) and is projected onto
        a new image with size (width=200, height=200), its new position will
        be (x1=20, y1=40). (Analogous for x2/y2.)

        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).

        Parameters
        ----------
        from_shape : tuple
            Shape of the original image. (Before resize.)

        to_shape : tuple
            Shape of the new image. (After resize.)

        Returns
        -------
        out : BoundingBox
            BoundingBox object with new coordinates.
        """
        if from_shape[:2] == to_shape[:2]:
            return self.copy()
        else:
            from_height, from_width = from_shape[:2]
            to_height, to_width = to_shape[:2]
            x1 = (self.x1 / from_width) * to_width
            x2 = (self.x2 / from_width) * to_width
            y1 = (self.y1 / from_height) * to_height
            y2 = (self.y2 / from_height) * to_height
            if x1 == x2:
                if x1 == 0:
                    x2 += 1
                else:
                    x1 -= 1
            if y1 == y2:
                if y1 == 0:
                    y2 += 1
                else:
                    y1 -= 1
            return self.copy(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2)

    def shift(self, top=None, right=None, bottom=None, left=None):
        top = top if top is not None else 0
        right = right if right is not None else 0
        bottom = bottom if bottom is not None else 0
        left = left if left is not None else 0
        return self.copy(
            x1=self.x1+left-right,
            x2=self.x2+left-right,
            y1=self.y1+top-bottom,
            y2=self.y2+top-bottom)

    def extend(self, all_sides=0, top=0, right=0, bottom=0, left=0):
        return BoundingBox(
            x1=self.x1-all_sides-left,
            x2=self.x2+all_sides+right,
            y1=self.y1-all_sides-top,
            y2=self.y2+all_sides+bottom)

    def intersection(self, other, default=None):
        """两个BoundingBox的交的面积"""
        eu.do_assert(isinstance(other, BoundingBox),
                     message="intersection@BoundingBox: other type error")
        x1_i = max(self.x1, other.x1)
        y1_i = max(self.y1, other.y1)
        x2_i = min(self.x2, other.x2)
        y2_i = min(self.y2, other.y2)
        if x1_i >= x2_i or y1_i >= y2_i:
            return default
        else:
            return BoundingBox(x1=x1_i, y1=y1_i, x2=x2_i, y2=y2_i)

    def union(self, other):
        """两个BoundinigBox的并的面积"""
        eu.do_assert(isinstance(other, BoundingBox),
                     message="union@BoundingBox: other type error")
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2))

    def iou(self, other):
        """两个BoundingBox之间的交/并的值"""
        inters = self.intersection(other)
        if inters is None:
            return 0
        else:
            return inters.area / self.union(other).area

    def is_fully_within_image(self, image):
        """当前的BoundingBox是不是在给定的图像内"""
        if isinstance(image, tuple):
            shape = image
        else:
            shape = image.shape
        height, width = shape[:2]
        b_width = self.x1 >= 0 and self.x2 <= width
        b_height = self.y1 >= 0 and self.y2 <= height
        return b_width and b_height

    def is_partly_within_image(self, image):
        """当前的BoundingBox是不是部分在给定的图像内"""
        if isinstance(image, tuple):
            shape = image
        else:
            shape = image.shape
        height, width = shape[:2]
        img_bb = BoundingBox(x1=0, x2=width, y1=0, y2=height)
        return self.intersection(img_bb) is not None

    def is_out_of_image(self, image, fully=True, partly=False):
        if self.is_fully_within_image(image):
            return False
        elif self.is_partly_within_image(image):
            return partly
        else:
            return fully

    def to_keypoints(self):
        return [
            KeyPoint(x=self.x1, y=self.y1),
            KeyPoint(x=self.x2, y=self.y1),
            KeyPoint(x=self.x2, y=self.y2),
            KeyPoint(x=self.x1, y=self.y2)
        ]

    def copy(self, x1=None, y1=None, x2=None, y2=None):
        return BoundingBox(
            x1=self.x1 if x1 is None else x1,
            x2=self.x2 if x2 is None else x2,
            y1=self.y1 if y1 is None else y1,
            y2=self.y2 if y2 is None else y2)


class BoundingBoxesOnImage(object):
    """
    Object that represents all bounding boxes on a single image.

    Parameters
    ----------
    bounding_boxes : list of BoundingBox
        List of bounding boxes on the image.

    shape : tuple of int
        The shape of the image on which the bounding boxes are placed.

    Examples
    --------
    >>> bbs = [
    >>>     BoundingBox(x1=10, y1=20, x2=20, y2=30),
    >>>     BoundingBox(x1=25, y1=50, x2=30, y2=70)
    >>> ]
    >>> bbs_oi = BoundingBoxesOnImage(bbs, shape=image.shape)
    """
    def __init__(self, bounding_boxes, shape):
        self.bounding_boxes = bounding_boxes
        if eu.is_np_array(shape):
            self.shape = shape.shape
        else:
            eu.do_assert(isinstance(shape, (tuple, list)))
            self.shape = tuple(shape)

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    def project(self, image):
        """
        Project bounding boxes from one image to a new one.

        Parameters
        ----------
        image : ndarray or tuple
            New image onto which the bounding boxes are to be projected.
            May also simply be that new image's shape tuple.

        Returns
        -------
        keypoints : BoundingBoxesOnImage
            Object containing all projected bounding boxes.
        """
        if eu.is_np_array(image):
            shape = image.shape
        else:
            shape = image

        if shape[:2] == self.shape[:2]:
            return self.deepcopy()
        else:
            bounding_boxes = [
                bb.project(self.shape, shape) for bb in self.bounding_boxes]
            return BoundingBoxesOnImage(bounding_boxes, shape)

    def shift(self, top=None, right=None, bottom=None, left=None):
        bbs_new = [bb.shift(top=top, right=right, bottom=bottom, left=left)
                   for bb in self.bounding_boxes]
        return BoundingBoxesOnImage(bbs_new, shape=self.shape)

    def remove_out_of_image(self, fully=True, partly=False):
        bbs_clean = [bb for bb in self.bounding_boxes
                     if not bb.is_out_of_image(self.shape, fully=fully, partly=partly)]
        return BoundingBoxesOnImage(bbs_clean, shape=self.shape)

    def deepcopy(self):
        return BoundingBoxesOnImage(self.bounding_boxes, self.shape)
