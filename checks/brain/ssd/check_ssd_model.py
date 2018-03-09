# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/9

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from optparse import OptionParser

from datum.utils.process_config import process_config
from eagle.brain.ssd.models.vgg import SSDVGG


parser = OptionParser()
parser.add_option("-c", "--conf",
                  dest="configure",
                  help="configure filename")
(options, args) = parser.parse_args()
if options.configure:
    conf_file = str(options.configure)
else:
    print('please sspecify --conf configure filename')
    exit(0)

common_params, dataset_params, net_params, solver_params, box_encoder_params = \
    process_config(conf_file)

net = SSDVGG(common_params, net_params, box_encoder_params)
images = tf.placeholder(dtype=tf.float32, shape=(32, 300, 300, 3))
model_spec = net.inference(images)
print(model_spec)
