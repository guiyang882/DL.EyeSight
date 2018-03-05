# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from abc import ABCMeta, abstractmethod


class SetMeta:

    def __init__(self, common_params, dataset_params):
        raise NotImplementedError

    def batch(self):
        raise NotImplementedError
