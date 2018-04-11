# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/4/3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
from shutil import copyfile
import numpy as np

from xml.dom.minidom import parseString
from lxml.etree import Element, SubElement, tostring
import pprint
import cv2

# label_id_map = {
#     ""
# }

def format_voc_string(filename, anno_infos):
    # anno_cell in anno_infos
    ## anno_cell is dict{"label": "car", "p1":[x1, y1], "p2":[x2, y2]}

    node_root = Element('annotation')

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = filename

    image = cv2.imread(filename)
    width, height = image.shape[:2]

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for anno_cell in anno_infos:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = anno_cell["label"]

        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'

        node_bndbox = SubElement(node_object, 'bndbox')
        node_x1 = SubElement(node_bndbox, 'x1')
        node_x1.text = str(anno_cell["p1"][0])
        node_y1 = SubElement(node_bndbox, 'y1')
        node_y1.text = str(anno_cell["p1"][1])

        node_x2 = SubElement(node_bndbox, 'x2')
        node_x2.text = str(anno_cell["p2"][0])
        node_y2 = SubElement(node_bndbox, 'y2')
        node_y2.text = str(anno_cell["p2"][1])

        node_x3 = SubElement(node_bndbox, 'x3')
        node_x3.text = str(anno_cell["p3"][0])
        node_y3 = SubElement(node_bndbox, 'y3')
        node_y3.text = str(anno_cell["p3"][1])

        node_x4 = SubElement(node_bndbox, 'x4')
        node_x4.text = str(anno_cell["p4"][0])
        node_y4 = SubElement(node_bndbox, 'y4')
        node_y4.text = str(anno_cell["p4"][1])

    xml = tostring(node_root, pretty_print=True)
    # dom = parseString(xml)
    return xml

data_prefix = "/Volumes/projects/DataSets/VEDIA/"
images_dir = ["512/Vehicules512/", "1024/Vehicules1024/"]
annotations_filepath = ["512/Annotations512/annotation512.txt"]

def convert():
    save_dir_prefix = "/Volumes/projects/DataSets/VEDIA/VOCFORMAT/"

    label_set = set()
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
                images_dict.setdefault(image_path, [])
                images_dict[image_path].append(line)

        for img_path in images_dict.keys():
            if not os.path.isfile(img_path):
                raise IOError("{} image path not found !".format(img_path))

            anno_infos = list()
            for line in images_dict[img_path]:
                anno_cell = dict()
                center_x, center_y = float(line[1]), float(line[2])
                rotate_theta = float(line[3])
                points = np.array(list(map(float, line[4:12])),
                                  np.int32).reshape((2, -1)).T
                fully_contain = int(line[-2])
                occluded = int(line[-1])
                if occluded:
                    continue
                # print(points.shape)
                label = line[-3]
                label_set.add(label)
                anno_cell["label"] = label
                anno_cell["p1"] = points[0, :]
                anno_cell["p2"] = points[1, :]
                anno_cell["p3"] = points[2, :]
                anno_cell["p4"] = points[3, :]
                anno_infos.append(anno_cell)
            if len(anno_infos) == 0:
                print(img_path)
                continue

            # copy file to dest
            image_name = img_path.split("/")[-1]
            copyfile(img_path, save_dir_prefix + "JPEGImages/" + image_name)
            voc_xml = format_voc_string(img_path, anno_infos)
            anno_name = img_path.split("/")[-1].replace("png", "xml")
            anno_file_path = save_dir_prefix + "Annotations/" + anno_name
            with open(anno_file_path, "wb") as writer:
                writer.write(voc_xml)
            # print(voc_xml)
            # return
    print(label_set)

if __name__ == '__main__':
    convert()