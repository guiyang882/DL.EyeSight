# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/6

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
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
  print('please specify --conf configure filename')
  exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)
print("After Proces Config File !")
# dataset = YoloDataSet(common_params, dataset_params)
# print("Prepared DataSet !")
net = YoloUNet(common_params, net_params)
print("Building the Deep Learning Model !")
solver = YoloUSolver(None, net, common_params, solver_params)
print("Now Start Learning Best Parameters !")
image_path = "/Volumes/projects/DataSets/CSUVideo/512x512/large_tunisia_total/JPEGImages/000011_1428_408_1940_920_35.jpg"

img_width, img_height = 512, 512
single_image = cv2.imread(image_path)
resized_img = cv2.resize(single_image, (img_height, img_width))
np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
np_img = np_img.astype(np.float32)
np_img = np_img / 255.0 * 2 - 1
np_img = np.reshape(np_img, (1, img_height, img_width, 3))

(xmin, ymin, xmax, ymax, class_num) = solver.model_predict(np_img)

cv2.rectangle(resized_img, (int(xmin), int(ymin)),
                      (int(xmax), int(ymax)), (0, 0, 255))
# cv2.imwrite('cat_out.jpg', resized_img)
cv2.imshow('cat_out.jpg', resized_img)
cv2.waitKey()
