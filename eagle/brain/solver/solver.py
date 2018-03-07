# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Solver(object):
    def __init__(self, dataset, net, common_params, solver_params):
        raise NotImplementedError

    def solve(self):
        raise NotImplementedError
