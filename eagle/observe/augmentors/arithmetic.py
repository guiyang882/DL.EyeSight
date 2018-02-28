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
from eagle.parameter import Deterministic, DiscreteUniform, Binomial


class Add(Augmentor):
    """
    Add a value to all pixels in an image.

    Parameters
    ----------
    value : int or iterable of two ints or StochasticParameter, optional(default=0)
        Value to add to all
        pixels.
            * If an int, then that value will be used for all images.
            * If a tuple (a, b), then a value from the discrete range [a .. b]
              will be used.
            * If a StochasticParameter, then a value will be sampled per image
              from that parameter.

    per_channel : bool or float, optional(default=False)
        Whether to use the same value for all channels (False)
        or to sample a new value for each channel (True).
        If this value is a float p, then for p percent of all images
        `per_channel` will be treated as True, otherwise as False.

    name : string, optional(default=None)
        See `Augmenter.__init__()`

    deterministic : bool, optional(default=False)
        See `Augmenter.__init__()`

    random_state : int or np.random.RandomState or None, optional(default=None)
        See `Augmenter.__init__()`

    Examples
    --------
    >>> aug = iaa.Add(10)

    always adds a value of 10 to all pixels in the image.

    >>> aug = iaa.Add((-10, 10))

    adds a value from the discrete range [-10 .. 10] to all pixels of
    the input images. The exact value is sampled per image.

    >>> aug = iaa.Add((-10, 10), per_channel=True)

    adds a value from the discrete range [-10 .. 10] to all pixels of
    the input images. The exact value is sampled per image AND channel,
    i.e. to a red-channel it might add 5 while subtracting 7 from the
    blue channel of the same image.

    >>> aug = iaa.Add((-10, 10), per_channel=0.5)

    same as previous example, but the `per_channel` feature is only active
    for 50 percent of all images.

    """

    def __init__(self, value=0, per_channel=False, name=None,
                 deterministic=False, random_state=None):
        super(Add, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        if eu.is_single_integer(value):
            eu.do_assert(-255 <= value <= 255,
                         "Expected value to have range [-255, 255], got value %d." % (value,))
            self.value = Deterministic(value)
        elif eu.is_iterable(value):
            eu.do_assert(len(value) == 2,
                         "Expected tuple/list with 2 entries, got %d entries." % (len(value),))
            self.value = DiscreteUniform(value[0], value[1])
        elif isinstance(value, StochasticParameter):
            self.value = value
        else:
            raise Exception("Expected float or int, tuple/list with 2 entries or StochasticParameter. Got %s." % (type(value),))

        if per_channel in [True, False, 0, 1, 0.0, 1.0]:
            self.per_channel = Deterministic(int(per_channel))
        elif eu.is_single_number(per_channel):
            eu.do_assert(0 <= per_channel <= 1.0,
                         "Expected bool, or number in range [0, 1.0] for per_channel, got %s." % (type(per_channel),))
            self.per_channel = Binomial(per_channel)
        else:
            raise Exception("Expected per_channel to be boolean or number or StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        input_dtypes = eu.copy_dtypes_for_restore(images)

        result = images
        nb_images = len(images)
        seeds = random_state.randint(0, 10**6, (nb_images,))
        for i in range(nb_images):
            image = images[i].astype(np.int32)
            rs_image = eu.new_random_state(seeds[i])
            per_channel = self.per_channel.draw_sample(random_state=rs_image)
            if per_channel == 1:
                nb_channels = image.shape[2]
                samples = self.value.draw_samples((nb_channels,), random_state=rs_image)
                for c, sample in enumerate(samples):
                    # TODO make value range more flexible
                    eu.do_assert(-255 <= sample <= 255)
                    image[..., c] += sample
            else:
                sample = self.value.draw_sample(random_state=rs_image)
                # TODO make value range more flexible
                eu.do_assert(-255 <= sample <= 255)
                image += sample
            result[i] = image

        # TODO make value range more flexible
        eu.clip_augmented_images_(result, 0, 255)
        eu.restore_augmented_images_dtypes_(result, input_dtypes)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return [self.value]
