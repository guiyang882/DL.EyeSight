# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/18

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from optparse import OptionParser

from datum.utils.process_config import process_config
from datum.models.ssd.ssd_dataset import SSDDataSet
from eagle.brain.ssd.models.vgg import SSDVGG
from eagle.brain.solver.ssd_solver import SSDSolver

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


data_generator = SSDDataSet(common_params, dataset_params, box_encoder_params)
net = SSDVGG(common_params, net_params, box_encoder_params)
solver = SSDSolver(data_generator, net, common_params, solver_params)
solver.solve()
