# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/20

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import xml.etree.ElementTree as ElementTree

import pickle as pkl
from PIL import Image
import numpy as np

train_set_01 = [('2007', 'train'), ('2012', 'train')]
train_set_02 = [('2012', 'val')]
val_set = [('2007', 'val')]
test_set = [('2007', 'test')]

classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

parser = argparse.ArgumentParser(
    description='Convert Pascal VOC 2007+2012 detection datum to Numpy.')
parser.add_argument(
    '-p',
    '--path_to_voc',
    help='path to VOCdevkit directory',
    default='/Volumes/projects/repos/VOC.SOURCE.DATA/VOCdevkit')


def get_boxes_for_id(voc_path, year, image_id):
    """Get object bounding boxes annotations for given image.

    Parameters
    ----------
    voc_path : str
        Path to VOCdevkit directory.
    year : str
        Year of datum containing image. Either '2007' or '2012'.
    image_id : str
        Pascal VOC identifier for given image.

    Returns
    -------
    boxes : array of int
        bounding box annotations of class label, xmin, ymin, xmax, ymax as a
        5xN array.
    """
    fname = os.path.join(voc_path, 'VOC{}/Annotations/{}.xml'.format(year, image_id))
    if not os.path.exists(fname):
        return None
    in_file = open(fname, 'r')
    xml_tree = ElementTree.parse(in_file)
    in_file.close()
    root = xml_tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        label = obj.find('name').text
        if label not in classes or int(
                difficult) == 1:  # exclude difficult or unlisted classes
            continue
        xml_box = obj.find('bndbox')
        bbox = (classes.index(label), int(xml_box.find('xmin').text),
                int(xml_box.find('ymin').text), int(xml_box.find('xmax').text),
                int(xml_box.find('ymax').text))
        boxes.extend(bbox)
    return np.array(boxes)  # .T  # return transpose so last dimension is variable length

def get_image_for_id(voc_path, year, image_id):
    """Get image data as uint8 array for given image.

    Parameters
    ----------
    voc_path : str
        Path to VOCdevkit directory.
    year : str
        Year of datum containing image. Either '2007' or '2012'.
    image_id : str
        Pascal VOC identifier for given image.

    Returns
    -------
    image_data : array of uint8
        Compressed JPEG byte string represented as array of uint8.
    """
    fname = os.path.join(voc_path, 'VOC{}/JPEGImages/{}.jpg'.format(year, image_id))
    if not os.path.exists(fname):
        return None
    data = Image.open(fname)
    img_matrix = np.asarray(data.convert('RGB'))
    data.close()
    return img_matrix

def get_ids(voc_path, datasets):
    ids = []
    for year, image_set in datasets:
        id_file = os.path.join(voc_path, 'VOC{}/ImageSets/Main/{}.txt'.format(year, image_set))
        if not os.path.exists(id_file):
            continue
        image_ids = open(id_file, 'r')
        ids.extend(map(str.strip, image_ids.readlines()))
        image_ids.close()
    return ids

def add_to_dataset(voc_path, years, ids):
    """Process all given ids and adds them to given datasets."""
    img_data_list = []
    img_boxes_list = []
    for i, voc_id in enumerate(ids):
        for year in years:
            image_data = get_image_for_id(voc_path, year, voc_id)
            image_boxes = get_boxes_for_id(voc_path, year, voc_id)
            if image_data is None or image_boxes is None:
                continue
            img_data_list.append(image_data)
            img_boxes_list.append(image_boxes)
    return img_data_list, img_boxes_list


def _main(args):
    voc_path = os.path.expanduser(args.path_to_voc)

    f_train_path_01 = os.path.join(voc_path, 'pascal_voc_07_12_train01.pkl')
    f_train_path_02 = os.path.join(voc_path, 'pascal_voc_07_12_train02.pkl')
    f_val_path = os.path.join(voc_path, 'pascal_voc_07_12_val.pkl')
    f_test_path = os.path.join(voc_path, 'pascal_voc_07_12_test.pkl')

    train_ids_01 = get_ids(voc_path, train_set_01)
    train_ids_02 = get_ids(voc_path, train_set_02)
    val_ids = get_ids(voc_path, val_set)
    test_ids = get_ids(voc_path, test_set)
    
    print('Processing Pascal VOC test set.')
    test_img_list, test_boxes_list = add_to_dataset(voc_path, ['2007'], test_ids)
    with open(f_test_path, "wb") as pkl_test_h:
        pkl.dump(dict(images=test_img_list, boxes=test_boxes_list), pkl_test_h)

    print('Processing Pascal VOC val set.')
    val_img_list, val_boxes_list = add_to_dataset(voc_path, ['2007', '2012'], val_ids)
    with open(f_val_path, "wb") as pkl_val_h:
        pkl.dump(dict(images=val_img_list, boxes=val_boxes_list), pkl_val_h)

    print('Processing Pascal VOC training set.')
    train_img_list, train_boxes_list = add_to_dataset(voc_path, ['2007', '2012'], train_ids_01)
    with open(f_train_path_01, "wb") as pkl_train_h_01:
        pkl.dump(dict(images=train_img_list, boxes=train_boxes_list), pkl_train_h_01)

    train_img_list, train_boxes_list = add_to_dataset(voc_path, ['2007', '2012'], train_ids_02)
    with open(f_train_path_02, "wb") as pkl_train_h_02:
        pkl.dump(dict(images=train_img_list, boxes=train_boxes_list), pkl_train_h_02)


if __name__ == '__main__':
    _main(parser.parse_args())
