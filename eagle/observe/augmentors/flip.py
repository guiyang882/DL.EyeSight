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
