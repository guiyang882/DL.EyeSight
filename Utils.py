# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/18

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np


def checkPath(path):
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            print(filename)

def get_Files_List(path):
    files_list=[]
    for path, subdirs,files in os.walk(path):
        for filename in files:
            files_list.append(os.path.join(path, filename))
    return files_list


def NMS(rects, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(rects) == 0:
        print("WARNING: Passed Empty Boxes Array")
        return []

    # initialize the list of picked indexes
    pick = []
    x1, x2, y1, y2, conf = [], [], [], [], []
    for rect in rects:
        x1.append(rect.x1)
        x2.append(rect.x2)
        y1.append(rect.y1)
        y2.append(rect.y2)
        conf.append(rect.true_confidence)
    # grab the coordinates of the bounding boxes
    x1 = np.array(x1)
    y1 = np.array(y1)
    x2 = np.array(x2)
    y2 = np.array(y2)
    conf = np.array(conf)
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(conf)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]
            # union = area[j] + float(w * h) - overlap

            # iou = overlap/union

            # if there is sufficient overlap, suppress the
            # current bounding box
            if (overlap > overlapThresh):
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    picked = []
    for i in pick: picked.append(rects[i])
    return picked