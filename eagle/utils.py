# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/26

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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


def do_assert(condition, message="Assertion Failed"):
    if not condition:
        raise AssertionError(str(message))


def is_np_array(val):
    return isinstance(val, np.ndarray)
