# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import codecs
import xml.etree.ElementTree as ET


data_dir = "/Volumes/projects/DataSets/CSUVideo/512x512"
image_sets = ["large_000013363_total", "large_000014631_total",
              "large_minneapolis_1_total", "large_tunisia_total"]

def parse_xml(xml_file):
    """
    Args:
      xml_file: the input xml file path

    Returns:
      image_path: string
      labels: list of [xmin, ymin, xmax, ymax, class]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    labels = []

    for item in root:
        if item.tag == 'object':
            obj_num = 1
            bndbox = item.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            labels.append([xmin, ymin, xmax, ymax, obj_num])

    return labels


def convert_list2str(labels):
    return ",".join([",".join(list(map(str, item))) for item in labels])


for dataset in image_sets:
    anno_prefix = "/".join([data_dir, dataset, "Annotations"])
    image_prefix = "/".join([data_dir, dataset, "JPEGImages"])
    with codecs.open(data_dir + "/" + dataset + ".txt", "w", "utf8") as writer:
        for anno_name in os.listdir(anno_prefix):
            if anno_name.startswith("."):
                continue
            anno_path = "/".join([anno_prefix, anno_name])
            image_name = anno_name.replace("xml", "jpg")
            image_path = "/".join([image_prefix, image_name])
            if not os.path.isfile(image_path):
                print("{} not found !".format(image_path))
            labels = parse_xml(anno_path)
            anno_info = convert_list2str(labels)
            writer.write("{},{}\n".format(image_path, anno_info))
