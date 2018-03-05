# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/26

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import numbers
import numpy as np

import cv2
from scipy import misc


CURRENT_RANDOM_STATE = np.random.RandomState(42)


def seed(seedval):
    CURRENT_RANDOM_STATE.seed(seedval)


def current_random_state():
    return CURRENT_RANDOM_STATE


def new_random_state(seed=None, fully_random=False):
    if seed is None:
        if not fully_random:
            seed = CURRENT_RANDOM_STATE.randint(0, 10 ** 6, 1)[0]
    return np.random.RandomState(seed)


def dummy_random_state():
    return np.random.RandomState(1)


def copy_random_state(random_state, force_copy=False):
    if random_state == np.random and not force_copy:
        return random_state
    else:
        rs_copy = dummy_random_state()
        orig_state = random_state.get_state()
        rs_copy.set_state(orig_state)
        return rs_copy


def forward_random_state(random_state):
    random_state.uniform()


def do_assert(condition, message="Assertion Failed"):
    if not condition:
        raise AssertionError(str(message))


def is_np_array(val):
    return isinstance(val, np.ndarray)


def is_iterable(val):
    return isinstance(val, (tuple, list))


def is_callable(val):
    if sys.version_info[0] == 3 and sys.version_info[1] <= 2:
        return hasattr(val, '__call__')
    else:
        return callable(val)


def is_string(val):
    return isinstance(val, str)


def is_single_integer(val):
    return isinstance(val, numbers.Integral)


def is_single_float(val):
    return isinstance(val, numbers.Real) and not is_single_integer(val)


def is_single_number(val):
    return isinstance(val, numbers.Real) or isinstance(val, numbers.Integral)


def is_integer_array(val):
    return is_np_array(val) and issubclass(val.dtype.type, numbers.Integral)


def copy_dtypes_for_restore(images):
    return images.dtype if is_np_array(images) else [image.dtype for image in images]

def restore_augmented_images_dtypes_(images, orig_dtypes):
    if is_np_array(images):
        images = images.astype(orig_dtypes)
    else:
        for i in range(len(images)):
            images[i] = images[i].astype(orig_dtypes[i])

def restore_augmented_images_dtypes(images, orig_dtypes):
    if is_np_array(images):
        images = np.copy(images)
    else:
        images = [np.copy(image) for image in images]
    return restore_augmented_images_dtypes_(images, orig_dtypes)

def clip_augmented_images_(images, minval, maxval):
    if is_np_array(images):
        np.clip(images, minval, maxval, out=images)
    else:
        for i in range(len(images)):
            np.clip(images[i], minval, maxval, out=images[i])

def clip_augmented_images(images, minval, maxval):
    if is_np_array(images):
        images = np.copy(images)
    else:
        images = [np.copy(image) for image in images]
    return clip_augmented_images_(images, minval, maxval)

# --------------------------------------------------------------------------------
# Basic Function about the Image Utils
# --------------------------------------------------------------------------------

def imresize_many_images(images, sizes=None, interpolation=None):
    """
    Resize many images to a specified size.

    Parameters
    ----------
    images : (N,H,W,C) ndarray
        Array of the images to resize.
        Expected to usually be of dtype uint8.

    sizes : iterable of two ints
        The new size in (height, width)
        tools.

    interpolation : None or string or int, optional(default=None)
        The interpolation to use during resize.
        If int, then expected to be one of:
            * cv2.INTER_NEAREST (nearest neighbour interpolation)
            * cv2.INTER_LINEAR (linear interpolation)
            * cv2.INTER_AREA (area interpolation)
            * cv2.INTER_CUBIC (cubic interpolation)
        If string, then expected to be one of:
            * "nearest" (identical to cv2.INTER_NEAREST)
            * "linear" (identical to cv2.INTER_LINEAR)
            * "area" (identical to cv2.INTER_AREA)
            * "cubic" (identical to cv2.INTER_CUBIC)
        If None, the interpolation will be chosen automatically. For size
        increases, area interpolation will be picked and for size decreases,
        linear interpolation will be picked.

    Returns
    -------
    result : (N,H',W',C) ndarray
        Array of the resized images.

    """
    s = images.shape
    do_assert(len(s) == 4, s)
    nb_images = s[0]
    im_height, im_width = s[1], s[2]
    nb_channels = s[3]
    height, width = sizes[0], sizes[1]

    if height == im_height and width == im_width:
        return np.copy(images)

    ip = interpolation
    do_assert(ip is None or ip in ["nearest", "linear", "area", "cubic", cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC])
    if ip is None:
        if height > im_height or width > im_width:
            ip = cv2.INTER_AREA
        else:
            ip = cv2.INTER_LINEAR
    elif ip in ["nearest", cv2.INTER_NEAREST]:
        ip = cv2.INTER_NEAREST
    elif ip in ["linear", cv2.INTER_LINEAR]:
        ip = cv2.INTER_LINEAR
    elif ip in ["area", cv2.INTER_AREA]:
        ip = cv2.INTER_AREA
    elif ip in ["cubic", cv2.INTER_CUBIC]:
        ip = cv2.INTER_CUBIC
    else:
        raise Exception("Invalid interpolation order")

    result = np.zeros((nb_images, height, width, nb_channels), dtype=np.uint8)
    for img_idx in range(nb_images):
        # TODO fallback to scipy here if image isn't uint8
        result_img = cv2.resize(images[img_idx], (width, height), interpolation=ip)
        if len(result_img.shape) == 2:
            result_img = result_img[:, :, np.newaxis]
        result[img_idx] = result_img
    return result


def imresize_single_image(image, sizes, interpolation=None):
    """
    Resizes a single image.

    Parameters
    ----------
    image : (H,W,C) ndarray or (H,W) ndarray
        Array of the image to resize.
        Expected to usually be of dtype uint8.

    sizes : iterable of two ints
        See `imresize_many_images()`.

    interpolation : None or string or int, optional(default=None)
        See `imresize_many_images()`.

    Returns
    -------
    out : (H',W',C) ndarray or (H',W') ndarray
        The resized image.

    """
    grayscale = False
    if image.ndim == 2:
        grayscale = True
        image = image[:, :, np.newaxis]
    do_assert(len(image.shape) == 3, image.shape)
    rs = imresize_many_images(image[np.newaxis, :, :, :], sizes, interpolation=interpolation)
    if grayscale:
        return np.squeeze(rs[0, :, :, 0])
    else:
        return rs[0, ...]


def draw_grid(images, rows=None, cols=None):
    """
    Converts multiple input images into a single image showing them in a grid.

    Parameters
    ----------
    images : (N,H,W,3) ndarray or iterable of (H,W,3) array
        The input images to convert to a grid.
        Expected to be RGB and have dtype uint8.

    rows : None or int, optional(default=None)
        The number of rows to show in the grid.
        If None, it will be automatically derived.

    cols : None or int, optional(default=None)
        The number of cols to show in the grid.
        If None, it will be automatically derived.

    Returns
    -------
    grid : (H',W',3) ndarray
        Image of the generated grid.

    """
    if is_np_array(images):
        do_assert(images.ndim == 4)
    else:
        do_assert(is_iterable(images) and is_np_array(images[0]) and images[0].ndim == 3)

    nb_images = len(images)
    do_assert(nb_images > 0)
    cell_height = max([image.shape[0] for image in images])
    cell_width = max([image.shape[1] for image in images])
    channels = set([image.shape[2] for image in images])
    do_assert(len(channels) == 1, "All images are expected to have the same number of channels, but got channel set %s with length %d instead." % (str(channels), len(channels)))
    nb_channels = list(channels)[0]
    if rows is None and cols is None:
        rows = cols = int(math.ceil(math.sqrt(nb_images)))
    elif rows is not None:
        cols = int(math.ceil(nb_images / rows))
    elif cols is not None:
        rows = int(math.ceil(nb_images / cols))
    do_assert(rows * cols >= nb_images)

    width = cell_width * cols
    height = cell_height * rows
    grid = np.zeros((height, width, nb_channels), dtype=np.uint8)
    cell_idx = 0
    for row_idx in range(rows):
        for col_idx in range(cols):
            if cell_idx < nb_images:
                image = images[cell_idx]
                cell_y1 = cell_height * row_idx
                cell_y2 = cell_y1 + image.shape[0]
                cell_x1 = cell_width * col_idx
                cell_x2 = cell_x1 + image.shape[1]
                grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image
            cell_idx += 1

    return grid

def show_grid(images, rows=None, cols=None):
    """
    Converts the input images to a grid image and shows it in a new window.

    This function wraps around scipy.misc.imshow(), which requires the
    `see <image>` command to work. On Windows systems, this tends to not be
    the case.

    Parameters
    ----------
    images : (N,H,W,3) ndarray or iterable of (H,W,3) array
        See `draw_grid()`.

    rows : None or int, optional(default=None)
        See `draw_grid()`.

    cols : None or int, optional(default=None)
        See `draw_grid()`.

    """
    grid = draw_grid(images, rows=rows, cols=cols)
    misc.imshow(grid)