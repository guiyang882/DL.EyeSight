# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/14

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from optparse import OptionParser

from datum.utils.process_config import process_config
from eagle.brain.yolo.yolo_u_net import YOLOUNet


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

common_params, dataset_params, net_params, solver_params = \
    process_config(conf_file)

net = YOLOUNet(common_params, net_params)
images = tf.placeholder(dtype=tf.float32, shape=(32, 512, 512, 3))
model_spec = net.inference(images)
print(model_spec)
