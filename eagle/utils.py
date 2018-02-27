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