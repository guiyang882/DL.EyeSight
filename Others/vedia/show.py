# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/12

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 主要是可视化带有旋转角度的图像中的标注

import os
import codecs

import cv2
import numpy as np

data_prefix = "/Volumes/projects/DataSets/VEDIA/"
images_dir = ["512/Vehicules512/", "1024/Vehicules1024/"]
annotations_filepath = ["512/Annotations512/annotation512.txt",
                        "1024/Annotations1024/annotation1024.txt"]


def show_image():
    for img_dir, anno_file in zip(images_dir, annotations_filepath):
        abs_img_dir = data_prefix + img_dir
        abs_anno_path = data_prefix + anno_file
        if not os.path.isfile(abs_anno_path):
            raise ValueError("{} file not found !".format(abs_anno_path))
        images_dict = {}
        with codecs.open(abs_anno_path, "r", "utf8") as reader:
            for line in reader:
                line = line.strip().split(' ')
                name_prefix = line[0] + "_co.png"
                image_path = abs_img_dir + name_prefix
                if image_path in images_dict.keys():
                    images_dict[image_path].append(line)
                else:
                    images_dict[image_path] = [line]
        for img_path in images_dict.keys():
            if not os.path.isfile(img_path):
                raise IOError("{} image path not found !".format(img_path))
            image = cv2.imread(img_path)

            for line in images_dict[img_path]:
                center_x, center_y = float(line[1]), float(line[2])
                rotate_theta = float(line[3])
                points = np.array(list(map(float, line[4:12])),
                                  np.int32).reshape((2, -1)).T
                fully_contain = int(line[-2])
                occluded = int(line[-1])

                points = points.reshape((-1, 1, 2))
                cv2.polylines(image, [points], True, color=(255, 0, 0))

            cv2.imshow("src", image)
            cv2.waitKey()


if __name__ == '__main__':
    show_image()