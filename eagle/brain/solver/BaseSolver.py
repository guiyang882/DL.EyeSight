# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class BaseSolver:
    __metaclass__ = ABCMeta

    def __init(self, dataset, net, common_params, solver_params):
        raise NotImplementedError

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def solve(self):
        raise NotImplementedError
