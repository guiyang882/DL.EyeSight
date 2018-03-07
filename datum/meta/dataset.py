# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class DataSet(object):
    def __init__(self, common_params, dataset_params):
        if not isinstance(common_params, dict):
            raise TypeError("common_params must be dict")
        if not isinstance(dataset_params, dict):
            raise TypeError("dataset_params must be dict")


    def batch(self):
        raise NotImplementedError
