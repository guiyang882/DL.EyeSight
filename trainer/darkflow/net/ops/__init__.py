# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/22

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .simple import *
from .convolution import *
from .baseop import HEADER, LINE

op_types = {
	'convolutional': convolutional,
	'conv-select': conv_select,
	'connected': connected,
	'maxpool': maxpool,
	'leaky': leaky,
	'dropout': dropout,
	'flatten': flatten,
	'avgpool': avgpool,
	'softmax': softmax,
	'identity': identity,
	'crop': crop,
	'local': local,
	'select': select,
	'route': route,
	'reorg': reorg,
	'conv-extract': conv_extract,
	'extract': extract
}

def op_create(*args):
	layer_type = list(args)[0].type
	return op_types[layer_type](*args)