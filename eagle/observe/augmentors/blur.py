# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/28

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2
import numpy as np
from scipy import ndimage

import eagle.utils as eu
from eagle.observe.base.meta import Augmentor
from eagle.parameter import StochasticParameter
from eagle.parameter import Deterministic, DiscreteUniform, Uniform


class GaussianBlur(Augmentor):
    """
    Augmenter to blur images using gaussian kernels.

    Examples
    --------
    >>> aug = iaa.GaussianBlur(sigma=1.5)

    blurs all images using a gaussian kernel with standard deviation 1.5.

    >>> aug = iaa.GaussianBlur(sigma=(0.0, 3.0))

    blurs images using a gaussian kernel with a random standard deviation
    from the range 0.0 <= x <= 3.0. The value is sampled per image.
    """

    def __init__(self, sigma=0, name=None, deterministic=False, random_state=None):
        super(GaussianBlur, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        if eu.is_single_number(sigma):
            self.sigma = Deterministic(sigma)
        elif eu.is_iterable(sigma):
            eu.do_assert(len(sigma) == 2,
                         "Expected tuple/list with 2 entries, got %d entries." % (len(sigma),))
            self.sigma = Uniform(sigma[0], sigma[1])
        elif isinstance(sigma, StochasticParameter):
            self.sigma = sigma
        else:
            raise Exception("Expected float, int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(sigma),))

        self.eps = 0.001 # epsilon value to estimate whether sigma is above 0

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.sigma.draw_samples((nb_images,), random_state=random_state)
        for i in range(nb_images):
            nb_channels = images[i].shape[2]
            sig = samples[i]
            if sig > 0 + self.eps:
                # note that while gaussian_filter can be applied to all channels
                # at the same time, that should not be done here, because then
                # the blurring would also happen across channels (e.g. red
                # values might be mixed with blue values in RGB)
                for channel in range(nb_channels):
                    result[i][:, :, channel] = ndimage.gaussian_filter(result[i][:, :, channel], sig)
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.sigma]


class AverageBlur(Augmentor):
    """
    Blur an image by computing simple means over neighbourhoods.

    Examples
    --------
    >>> aug = iaa.AverageBlur(k=5)

    Blurs all images using a kernel size of 5x5.

    >>> aug = iaa.AverageBlur(k=(2, 5))

    Blurs images using a varying kernel size per image, which is sampled
    from the interval [2..5].

    >>> aug = iaa.AverageBlur(k=((5, 7), (1, 3)))

    Blurs images using a varying kernel size per image, which's height
    is sampled from the interval [5..7] and which's width is sampled
    from [1..3].
    """

    def __init__(self, k=1, name=None, deterministic=False, random_state=None):
        super(AverageBlur, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.mode = "single"
        if eu.is_single_number(k):
            self.k = Deterministic(int(k))
        elif eu.is_iterable(k):
            eu.do_assert(len(k) == 2)
            if all([eu.is_single_number(ki) for ki in k]):
                self.k = DiscreteUniform(int(k[0]), int(k[1]))
            elif all([isinstance(ki, StochasticParameter) for ki in k]):
                self.mode = "two"
                self.k = (k[0], k[1])
            else:
                k_tuple = [None, None]
                if eu.is_single_number(k[0]):
                    k_tuple[0] = Deterministic(int(k[0]))
                elif eu.is_iterable(k[0]) and all(
                        [eu.is_single_number(ki) for ki in k[0]]):
                    k_tuple[0] = DiscreteUniform(int(k[0][0]), int(k[0][1]))
                else:
                    raise Exception("k[0] expected to be int or tuple of two ints, got %s" % (type(k[0]),))

                if eu.is_single_number(k[1]):
                    k_tuple[1] = Deterministic(int(k[1]))
                elif eu.is_iterable(k[1]) and all(
                        [eu.is_single_number(ki) for ki in k[1]]):
                    k_tuple[1] = DiscreteUniform(int(k[1][0]), int(k[1][1]))
                else:
                    raise Exception("k[1] expected to be int or tuple of two ints, got %s" % (type(k[1]),))

                self.mode = "two"
                self.k = k_tuple
        elif isinstance(k, StochasticParameter):
            self.k = k
        else:
            raise Exception("Expected int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(k),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        if self.mode == "single":
            samples = self.k.draw_samples((nb_images,), random_state=random_state)
            samples = (samples, samples)
        else:
            samples = (
                self.k[0].draw_samples((nb_images,), random_state=random_state),
                self.k[1].draw_samples((nb_images,), random_state=random_state),
            )
        for i in range(nb_images):
            kh, kw = samples[0][i], samples[1][i]
            #print(images.shape, result.shape, result[i].shape)
            kernel_impossible = (kh == 0 or kw == 0)
            kernel_does_nothing = (kh == 1 and kw == 1)
            if not kernel_impossible and not kernel_does_nothing:
                image_aug = cv2.blur(result[i], (kh, kw))
                # cv2.blur() removes channel axis for single-channel images
                if image_aug.ndim == 2:
                    image_aug = image_aug[..., np.newaxis]
                result[i] = image_aug
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.k]


class MedianBlur(Augmentor):
    """
    Blur an image by computing median values over neighbourhoods.

    Median blurring can be used to remove small dirt from images.
    At larger kernel sizes, its effects have some similarity with Superpixels.

    Examples
    --------
    >>> aug = iaa.MedianBlur(k=5)

    blurs all images using a kernel size of 5x5.

    >>> aug = iaa.MedianBlur(k=(3, 7))

    blurs images using a varying kernel size per image, which is
    and odd value sampled from the interval [3..7], i.e. 3 or 5 or 7.
    """

    def __init__(self, k=1, name=None, deterministic=False, random_state=None):
        super(MedianBlur, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        if eu.is_single_number(k):
            eu.do_assert(k % 2 != 0,
                         "Expected k to be odd, got %d. Add or subtract 1." % (int(k),))
            self.k = Deterministic(int(k))
        elif eu.is_iterable(k):
            eu.do_assert(len(k) == 2)
            eu.do_assert(all([eu.is_single_number(ki) for ki in k]))
            eu.do_assert(k[0] % 2 != 0,
                         "Expected k[0] to be odd, got %d. Add or subtract 1." % (int(k[0]),))
            eu.do_assert(k[1] % 2 != 0,
                         "Expected k[1] to be odd, got %d. Add or subtract 1." % (int(k[1]),))
            self.k = DiscreteUniform(int(k[0]), int(k[1]))
        elif isinstance(k, StochasticParameter):
            self.k = k
        else:
            raise Exception("Expected int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(k),))

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.k.draw_samples((nb_images,), random_state=random_state)
        for i in range(nb_images):
            ki = samples[i]
            if ki > 1:
                ki = ki + 1 if ki % 2 == 0 else ki
                image_aug = cv2.medianBlur(result[i], ki)
                # cv2.medianBlur() removes channel axis for single-channel
                # images
                if image_aug.ndim == 2:
                    image_aug = image_aug[..., np.newaxis]
                result[i] = image_aug
        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.k]
