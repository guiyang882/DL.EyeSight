# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/28

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import eagle.utils as eu
from eagle.observe.base.meta import Augmentor
from eagle.parameter import StochasticParameter, Binomial


class Fliplr(Augmentor):
    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        super(Fliplr, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        if eu.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p type StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in range(nb_images):
            if samples[i] == 1:
                images[i] = np.fliplr(images[i])
        return images

    def _augment_keypoints(self,
                           keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images, ), random_state=random_state)
        for i, kps_oi in enumerate(keypoints_on_images):
            if samples[i] == 1:
                width = keypoints_on_images.shape[1]
                for kp in keypoints_on_images.keypoints:
                    kp.x = (width - 1) - kp.x
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]


class Flipud(Augmentor):
    def __init__(self, p=0, name=None, deterministic=False, random_state=None):
        super(Flipud, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)
        if eu.is_single_number(p):
            self.p = Binomial(p)
        elif isinstance(p, StochasticParameter):
            self.p = p
        else:
            raise Exception("Expected p type StochasticParameter")

    def _augment_images(self, images, random_state, parents, hooks):
        nb_images = len(images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i in range(nb_images):
            if samples[i] == 1:
                images[i] = np.flipud(images[i])
        return images

    def _augment_keypoints(self,
                           keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        samples = self.p.draw_samples((nb_images,), random_state=random_state)
        for i, kps_oi in enumerate(keypoints_on_images):
            if samples[i] == 1:
                height = keypoints_on_images.shape[0]
                for kp in keypoints_on_images.keypoints:
                    kp.y = (height - 1) - kp.y
        return keypoints_on_images

    def get_parameters(self):
        return [self.p]
