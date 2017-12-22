# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/22

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import cv2
import os


labels20 = ["aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor"]

# 8, 14, 15, 19

voc_models = ['yolo-full', 'yolo-tiny', 'yolo-small',  # <- v1
              'yolov1', 'tiny-yolov1', # <- v1.1 
              'tiny-yolo-voc', 'yolo-voc'] # <- v2

coco_models = ['tiny-coco', 'yolo-coco',  # <- v1.1
               'yolo', 'tiny-yolo'] # <- v2

coco_names = 'coco.names'
nine_names = '9k.names'

def labels(meta, FLAGS):    
    model = os.path.basename(meta['name'])
    if model in voc_models: 
        print("Model has a VOC model name, loading VOC labels.")
        meta['labels'] = labels20
    else:
        file = FLAGS.labels
        if model in coco_models:
            print("Model has a coco model name, loading coco labels.")
            file = os.path.join(FLAGS.config, coco_names)
        elif model == 'yolo9000':
            print("Model has name yolo9000, loading yolo9000 labels.")
            file = os.path.join(FLAGS.config, nine_names)
        with open(file, 'r') as f:
            meta['labels'] = list()
            labs = [l.strip() for l in f.readlines()]
            for lab in labs:
                if lab == '----': break
                meta['labels'] += [lab]
    if len(meta['labels']) == 0: 
        meta['labels'] = labels20

def is_inp(self, name): 
    return name.lower().endswith(('.jpg', '.jpeg', '.png'))

def show(im, allobj, S, w, h, cellx, celly):
    for obj in allobj:
        a = obj[5] % S
        b = obj[5] // S
        cx = a + obj[1]
        cy = b + obj[2]
        centerx = cx * cellx
        centery = cy * celly
        ww = obj[3]**2 * w
        hh = obj[4]**2 * h
        cv2.rectangle(im,
            (int(centerx - ww/2), int(centery - hh/2)),
            (int(centerx + ww/2), int(centery + hh/2)),
            (0,0,255), 2)
    cv2.imshow('result', im)
    cv2.waitKey()
    cv2.destroyAllWindows()

def show2(im, allobj):
    for obj in allobj:
        cv2.rectangle(im,
            (obj[1], obj[2]), 
            (obj[3], obj[4]), 
            (0,0,255),2)
    cv2.imshow('result', im)
    cv2.waitKey()
    cv2.destroyAllWindows()


_MVA = .05

def profile(self, net):
    pass