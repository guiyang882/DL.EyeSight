# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/6

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from optparse import OptionParser

from datum.utils.process_config import process_config
from datum.meta.text_dataset import TextDataSet
from eagle.brain.yolo.yolo_tiny_net import YoloTinyNet
from eagle.brain.solver.yolo_solver import YoloSolver


parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",  
                  help="configure filename")
(options, args) = parser.parse_args() 
if options.configure:
  conf_file = str(options.configure)
else:
  print('please sspecify --conf configure filename')
  exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)
dataset = TextDataSet(common_params, dataset_params)
net = YoloTinyNet(common_params, net_params)
solver = YoloSolver(dataset, net, common_params, solver_params)
solver.solve()