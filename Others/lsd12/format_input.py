# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/1/3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

p1 = "/".join(os.path.abspath(__file__).split("/")[:-3])
sys.path.insert(0, p1)
p1 = "/".join(os.path.abspath(__file__).split("/")[:-2])
sys.path.insert(0, p1)
p1 = "/".join(os.path.abspath(__file__).split("/")[:-1])
sys.path.insert(0, p1)

from importlib import reload
reload(sys)

import cv2

from datum.utils import tools
from Others.lsd12 import label_config as config


nwpu_voc_dir = "/Volumes/projects/repos/RSI/NWPUVHR10/NWPUVOCFORMAT/"
nwpu_voc_image_dir = nwpu_voc_dir + "JPEGImages/"
nwpu_voc_anno_dir = nwpu_voc_dir + "Annotations/"

vedia_voc_dir = "/Volumes/projects/repos/RSI/VEDAI/VEDIAVOCFORAMT/"
vedia_voc_image_dir = vedia_voc_dir + "JPEGImages/"
vedia_voc_anno_dir = vedia_voc_dir + "Annotations/"

lsd_voc_dir = "/Volumes/projects/repos/RSI/LSD10/"
lsd_voc_image_dir = lsd_voc_dir + "JPEGImages/"
lsd_voc_anno_dir = lsd_voc_dir + "Annotations/"


# 先确定每个原始数据集中的训练集和测试集
def split_dataset():
    nwpu_img_list = os.listdir(nwpu_voc_image_dir)
    vedia_img_list = os.listdir(vedia_voc_image_dir)
    test_nwpu_img_list = tools.rand_selected_file(nwpu_img_list)
    test_vedia_img_list = tools.rand_selected_file(vedia_img_list)
    with open(nwpu_voc_dir+"test.txt", "w") as test_nwpu_writer:
        for item in test_nwpu_img_list:
            test_nwpu_writer.write("{}\n".format(item))
    with open(nwpu_voc_dir+"train.txt", "w") as train_nwpu_writer:
        for item in nwpu_img_list:
            if item not in test_nwpu_img_list:
                train_nwpu_writer.write("{}\n".format(item))
    with open(vedia_voc_dir+"test.txt", "w") as test_vedia_writer:
        for item in test_vedia_img_list:
            test_vedia_writer.write("{}\n".format(item))
    with open(vedia_voc_dir+"train.txt", "w") as train_vedia_writer:
        for item in vedia_img_list:
            if item not in test_vedia_img_list:
                train_vedia_writer.write("{}\n".format(item))


# 更新数据集中的label信息
def flush_dataset():
    for anno_name in os.listdir(lsd_voc_anno_dir):
        abs_anno_path = lsd_voc_anno_dir + anno_name
        print(abs_anno_path)
        anno_targets = tools.extract_target_from_xml(abs_anno_path)
        new_anno_targets = list()
        for anno_info in anno_targets:
            label_name = anno_info[-1]
            label_id = config.sign_idx_dict[label_name]
            label_name = config.idx_sign_dict[label_id]
            new_anno_info = anno_info[:-1] + [label_name]
            new_anno_targets.append(new_anno_info)
        src_image = cv2.imread(
            lsd_voc_image_dir+anno_name.replace("xml", "jpg"))
        xml_obj = tools.fetch_xml_format(
            src_image, anno_name.replace("xml", "jpg"), new_anno_targets)
        with open(lsd_voc_anno_dir+anno_name, "w") as writer:
            writer.write(xml_obj)

# 获取标准的目标的label
def get_true_label_name(label_name):
    label_id = config.sign_idx_dict[label_name]
    label_name = config.idx_sign_dict[label_id]
    return label_name


# 将图像中非指定尺度数据进行标准化
def format_corp_images():
    for anno_name in os.listdir(lsd_voc_anno_dir):
        abs_anno_path = lsd_voc_anno_dir + anno_name
        abs_img_path = lsd_voc_image_dir + anno_name.replace("xml", "jpg")
        image_name = anno_name.replace("xml", "jpg")
        src_image = cv2.imread(abs_img_path)
        if src_image.shape == (512, 512, 3):
            continue

        h, w = src_image.shape[:2]
        if h <= 512 and w <= 512:
            continue
        
        print(abs_img_path)
        anno_targets = tools.extract_target_from_xml(abs_anno_path)
        new_anno_targets = list()
        for anno_info in anno_targets:
            label_name = get_true_label_name(anno_info[-1])
            new_anno_info = anno_info[:-1] + [label_name]
            new_anno_targets.append(new_anno_info)
        crop_list, anno_list = tools.crop_samples(src_image, new_anno_targets)

        for i in range(len(crop_list)):
            x0, y0, x1, y1 = crop_list[i]
            # roi = im[y1:y2, x1:x2] opencv中类似NUMPY的裁剪
            sub_img = src_image[y0:y1, x0:x1]
            f_name = image_name[:-4] + "_%d_%d_%d_%d_%d.jpg" % (x0, y0, x1, y1, i)
            cv2.imwrite(lsd_voc_image_dir + f_name, sub_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            a_name = image_name[:-4]+ "_%d_%d_%d_%d_%d.xml" % (x0, y0, x1, y1, i)
            xml_obj = tools.fetch_xml_format(src_image, f_name, anno_list[i], "LSD12")
            with open(lsd_voc_anno_dir + a_name, "w") as writer:
                writer.write(xml_obj)

        os.remove(abs_img_path)
        os.remove(abs_anno_path)

# 根据图像文件列表，对数据集进行切分
from sklearn.model_selection import train_test_split

def split_train_valid_test():
    save_dir = "/Volumes/projects/repos/RSI/LSD10/"
    file_path = save_dir + "total.txt"
    image_list = list()
    with open(file_path, "r") as h:
        for line in h:
            line = line.strip()
            image_list.append(line)
    X_train, X_test = train_test_split(image_list, test_size=0.3, random_state=42)
    print(len(X_train), len(X_test))
    X_train, X_valid = train_test_split(X_train, test_size=0.2, random_state=42)
    print(len(X_train), len(X_valid))
    with open(save_dir+"train.txt", "w") as h1:
        for line in X_train:
            h1.write("{}\n".format(line))
    with open(save_dir+"valid.txt", "w") as h2:
        for line in X_valid:
            h2.write("{}\n".format(line))
    with open(save_dir+"test.txt", "w") as h3:
        for line in X_test:
            h3.write("{}\n".format(line))


if __name__ == '__main__':
    split_train_valid_test()
    # Others.show_targets(lsd_voc_image_dir, lsd_voc_anno_dir)
    pass
