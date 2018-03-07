# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class DataSet(object):
    """Base DataSet
  """

    def __init__(self, common_params, dataset_params):
        """
    common_params: A params dict 
    dataset_params: A params dict
    """
        raise NotImplementedError

    def batch(self):
        """Get batch
    """
        raise NotImplementedError
