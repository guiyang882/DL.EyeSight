# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/1/3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import cv2

import numpy as np
import xml.dom.minidom
import random
from lxml.etree import Element, SubElement, tostring

# 提取图像对应的标注的数据
def fetch_anno_targets_info(abs_anno_path, is_label_text=False):
    if not os.path.exists(abs_anno_path):
        raise IOError("No Such annotation file !")
    with open(abs_anno_path, "r") as anno_reader:
        total_annos = list()
        for line in anno_reader:
            line = line.strip()
            sub_anno = re.split("\(|\,|\)", line)
            a = [int(item) for item in sub_anno if len(item)]
            if len(a) == 5:
                if is_label_text:
                    total_annos.append(a[:4]+[config.idx_sign_dict[a[-1]]])
                else:
                    total_annos.append(a)
        return total_annos

def fetch_xml_format(src_img_data, f_name, anno_list, dataset):
    img_height, img_width, img_channle = src_img_data.shape

    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = dataset
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = f_name

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(img_width)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(img_height)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = str(img_channle)

    for anno_target in anno_list:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = anno_target[-1]
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(1 if anno_target[0]<0 else anno_target[0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(1 if anno_target[1]<0 else anno_target[1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(img_width-1 if anno_target[2]>=img_width else anno_target[2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(img_height-1 if anno_target[3]>=img_height else anno_target[3])
    xml_obj = tostring(node_root, pretty_print=True)
    xml_obj = xml_obj.decode("utf8")
    return xml_obj

# 给定一个标记文件，找到对应的目标的位置信息
def extract_target_from_xml(filename):
    if not os.path.exists(filename):
        raise IOError(filename + " not exists !")
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(filename)
    collection = DOMTree.documentElement
    # 获取集合中所有的目标
    targets = collection.getElementsByTagName("object")
    res = []
    for target in targets:
        target_name = target.getElementsByTagName('name')[0].childNodes[0].data
        bndbox = target.getElementsByTagName("bndbox")[0]
        xmin = bndbox.getElementsByTagName("xmin")[0].childNodes[0].data
        ymin = bndbox.getElementsByTagName("ymin")[0].childNodes[0].data
        xmax = bndbox.getElementsByTagName("xmax")[0].childNodes[0].data
        ymax = bndbox.getElementsByTagName("ymax")[0].childNodes[0].data
        res.append([int(xmin), int(ymin), int(xmax), int(ymax), target_name])
    return res

# 原始数据中多目标的显示
def show_targets(img_dir, anno_dir):
    for img_name in os.listdir(img_dir):
        if img_name.startswith("._"):
            continue
        abs_img_path = img_dir+img_name
        abs_anno_path = anno_dir+img_name.replace("jpg", "xml")
        target_annos = extract_target_from_xml(abs_anno_path)
        image = cv2.imread(abs_img_path)
        for target_info in target_annos:
            xmin, ymin, xmax, ymax = target_info[:4]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.imshow("src", image)
        cv2.waitKey()

# 采用蓄水池采样算法对序列进行采样
def rand_selected_file(file_list, K_ratio=2/7):
    K = int(len(file_list) * K_ratio)
    res = list()
    for i in range(0, len(file_list)):
        if i < K:
            res.append(file_list[i])
        else:
            M = random.randint(0, i)
            if M < K:
                res[M] = file_list[i]
    return res

def calc_rgb_mean():
    r_list, g_list, b_list = list(), list(), list()
    with open("/Volumes/projects/repos/RSI/LSD10/total.txt", "r") as reader:
        for line in reader.readlines():
            line = line.strip()
            src_img = cv2.imread(line)
            b, g, r = cv2.split(src_img)
            b_list.append(np.mean(b))
            g_list.append(np.mean(g))
            r_list.append(np.mean(r))
    print(np.mean(r_list))
    print(np.mean(g_list))
    print(np.mean(b_list))
"""
104.480289006
107.307103097
95.8043901467
"""

# 从样本中裁剪出制定的大小的候选样本，这其中必须要包含相应的目标
def crop_samples(src_image, anno_targets, SSD_IMG_W=512, SSD_IMG_H=512):

    def _crop_valid(area, anno_targets):
        anno_res = []
        for info in anno_targets:
            if ((info[0] >= area[0] and info[1] >= area[1]) and
                (info[2] <= area[2] and info[3] <= area[3])):
                anno_res.append(
                    [info[0] - area[0], info[1] - area[1],
                     info[2] - area[0], info[3] - area[1],
                     info[-1]])
            if (info[0] >= area[0] and info[1] >= area[1] and
                info[0] < area[2] and info[1] < area[3] and
                (not (info[2] <= area[2] and info[3] <= area[3]))):
                base = (info[2] - info[0]) * (info[3] - info[1])
                x_max_min = min(info[2], area[2])
                y_max_min = min(info[3], area[3])
                new_square = (x_max_min - info[0]) * (y_max_min - info[1])
                if new_square / base >= 0.8:
                    anno_res.append(
                        [info[0] - area[0], info[1] - area[1],
                         x_max_min - area[0], y_max_min - area[1],
                         info[-1]])
        return anno_res

    def _random_crop_for_target():
        img_height, img_width = src_image.shape[:2]
        crop_list, anno_list = [], []
        for idx in range(0, len(anno_targets)):
            c_x = (anno_targets[idx][0] + anno_targets[idx][2]) // 2
            c_y = (anno_targets[idx][1] + anno_targets[idx][3]) // 2

            u_x = random.randint(max(0, c_x - SSD_IMG_W // 2), anno_targets[idx][0])
            u_y = random.randint(max(0, c_y - SSD_IMG_H // 2), anno_targets[idx][1])

            area = [u_x, u_y, u_x + SSD_IMG_W, u_y + SSD_IMG_H]
            # 检测当前的候选框中是否包含了目标，并算出目标在给定图像的位置
            trans_targets = _crop_valid(area, anno_targets)
            if trans_targets:
                crop_list.append(area)
                anno_list.append(trans_targets)
        return crop_list, anno_list

    def _align_crop_for_target():
        h, w = src_image.shape[:2]
        crop_list, anno_list = [], []
        for lx in range(0, max(1, w-SSD_IMG_W+1), SSD_IMG_W//5):
            for ly in range(0, max(1, h-SSD_IMG_H+1), SSD_IMG_H//5):
                u_x, u_y = lx, ly
                # if lx + SSD_IMG_W > w:
                #     u_x = w - SSD_IMG_W
                # if ly + SSD_IMG_H > h:
                #     u_y = h - SSD_IMG_H
                area = [u_x, u_y, u_x + SSD_IMG_W, u_y + SSD_IMG_H]
                trans_targets = list()
                trans_targets = _crop_valid(area, anno_targets)
                if trans_targets:
                    crop_list.append(area)
                    anno_list.append(trans_targets)
        return crop_list, anno_list

    crop_list, anno_list = _align_crop_for_target()
    return crop_list, anno_list


if __name__ == '__main__':
    a = fetch_anno_targets_info(
        "/Volumes/projects/repos/RSI/NWPUVHR10/sub_annotation/001.txt")
    print(a)