# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/6

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from optparse import OptionParser

from datum.utils.process_config import process_config
from datum.meta.DetectDataSet import DetectDataSet
from eagle.brain.yolo.TinyYoloNet import TinyYoloNet
from eagle.brain.solver.YoloSolver import YoloSolver


parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure", help="configure file path")

(options, args) = parser.parse_args()
if options.configure:
    conf_file = str(options.configure)
    if not os.path.isfile(conf_file):
        raise ValueError("{} not found !".format(conf_file))
else:
    print("Please Enter configure file path !")
    exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)
dataset = DetectDataSet(common_params, dataset_params)
net = TinyYoloNet(common_params, net_params)
solver = YoloSolver(dataset, net, common_params, solver_params)
solver.solve()
