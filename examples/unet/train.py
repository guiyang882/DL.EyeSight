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
# from datum.models.yolo.yolo_dataset import YoloDataSet
from datum.models.yolo.yolo_batch_dataset import YoloDataSet
from eagle.brain.solver.yolo_u_solver import YoloUSolver
from eagle.brain.yolo.yolo_u_net import YoloUNet

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
print("After Proces Config File !")
dataset = YoloDataSet(common_params, dataset_params)
print("Prepared DataSet !")
net = YoloUNet(common_params, net_params)
print("Building the Deep Learning Model !")
solver = YoloUSolver(dataset, net, common_params, solver_params)
print("Now Start Learning Best Parameters !")
solver.solve()
