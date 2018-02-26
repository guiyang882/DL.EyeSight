# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/26

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
import numpy as np

import eagle.utils as eu


class Augmentor:
    __metaclass__ = ABCMeta
    
    def __init__(self, name=None, deterministic=False, random_state=None):
        """
        Create a new Augmenter instance.

        Parameters
        ----------
        name : None or string, optional(default=None)
            Name given to an Augmenter object. This name is used in print()
            statements as well as find and remove functions.
            If None, `UnnamedX` will be used as the name, where X is the
            Augmenter's class name.

        deterministic : bool, optional(default=False)
            Whether the augmenter instance's random state will be saved before
            augmenting images and then reset to that saved state after an
            augmentation (of multiple images/keypoints) is finished.
            I.e. if set to True, each batch of images will be augmented in the
            same way (e.g. first image might always be flipped horizontally,
            second image will never be flipped etc.).
            This is useful when you want to transform multiple batches of images
            in the same way, or when you want to augment images and keypoints
            on these images.
            Usually, there is no need to set this variable by hand. Instead,
            instantiate the augmenter with the defaults and then use
            `augmenter.to_deterministic()`.

        random_state : None or int or np.random.RandomState, optional(default=None)
            The random state to use for this
            augmenter.
                * If int, a new np.random.RandomState will be created using this
                  value as the seed.
                * If np.random.RandomState instance, the instance will be used directly.
                * If None, imgaug's default RandomState will be used, which's state can
                  be controlled using imgaug.seed(int).
            Usually there is no need to set this variable by hand. Instead,
            instantiate the augmenter with the defaults and then use
            `augmenter.to_deterministic()`.

        """
        super(Augmentor, self).__init__()
        self.name = name
        if name is None:
            self.name = "Unnamed" + self.__class__.__name__
        self.deterministic = deterministic
        if random_state is None:
            if self.deterministic:
                self.random_state = eu.new_random_state()
            else:
                self.random_state = eu.current_random_state()
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
        
        self.activated = True
        
    def augment_batches(self, batches):
        """
        Augment multiple batches of images.

        Parameters
        ----------
        batches : list
            List of image batches to augment.
            The expected input is a list, with each entry having one of the
            following datatypes:
                * ia.Batch
                * []
                * list of ia.KeypointsOnImage
                * list of (H,W,C) ndarray
                * list of (H,W) ndarray
                * (N,H,W,C) ndarray
                * (N,H,W) ndarray
            where N = number of images, H = height, W = width,
            C = number of channels.
            Each image is recommended to have dtype uint8 (range 0-255).

        Yields
        -------
        augmented_batch : ia.Batch or list of ia.KeypointsOnImage or list of (H,W,C) ndarray or list of (H,W) ndarray or (N,H,W,C) ndarray or (N,H,W) ndarray
            Augmented images/keypoints.
            Datatype usually matches the input datatypes per list element.
        """
        eu.do_assert(isinstance(batches, list))
        eu.do_assert(len(batches) > 0)

        batches_normalized = []
        batches_original_dts = []