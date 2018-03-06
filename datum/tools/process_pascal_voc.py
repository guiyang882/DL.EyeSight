# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import struct
import numpy as np
import xml.etree.ElementTree as ET


classes_name = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train","tvmonitor"
]

classes_num = {
    'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
    'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10,
    'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14, 'pottedplant': 15,
    'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19
}

DATA_ROOT = "/Volumes/projects/DataSets/VOC"
DATA_PATH = os.path.join(DATA_ROOT, "VOCdevkit/")
OUTPUT_PATH = os.path.join(DATA_ROOT, "pascal_voc_{}.txt")


def parse_xml(xml_file, year=2007):
    """
    Args:
      xml_file: the input xml file path

    Returns:
      image_path: string
      labels: list of [xmin, ymin, xmax, ymax, class]
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_path = ''
    labels = []

    for item in root:
        if item.tag == 'filename':
            if year == 2007:
                image_path = os.path.join(
                    DATA_PATH, 'VOC2007/JPEGImages', item.text)
            if year == 2012:
                image_path = os.path.join(
                    DATA_PATH, 'VOC2012/JPEGImages', item.text)
        elif item.tag == 'object':
            obj_name = item[0].text
            obj_num = classes_num[obj_name]
            bndbox = item.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            labels.append([xmin, ymin, xmax, ymax, obj_num])

    return image_path, labels


def convert_to_string(image_path, labels):
    out_string = ''
    out_string += image_path
    for label in labels:
        for i in label:
            out_string += ' ' + str(i)
    out_string += '\n'

    return out_string


def run_main(year=2007):
    print("Start format voc {} data !".format(year))
    out_file = open(OUTPUT_PATH.format(year), "w")
    if year == 2007:
        xml_dir = os.path.join(DATA_PATH, "VOC2007/Annotations/")
    if year == 2012:
        xml_dir = os.path.join(DATA_PATH, "VOC2012/Annotations/")

    xml_list = os.listdir(xml_dir)

    xml_list = [xml_dir + tmp for tmp in xml_list]
    for xml in xml_list:
        if not os.path.isfile(xml):
            print("{} not xml file path.".format(xml))
        image_path, labels = parse_xml(xml, year=year)
        record = convert_to_string(image_path, labels)
        out_file.write(record)
    out_file.close()

if __name__ == '__main__':
    run_main(year=2007)
    run_main(year=2012)
