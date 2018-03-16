# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 使用聚类方法进行目标框的聚类操作
import os
import numpy as np

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from Others.satellite.process import parse_xml


data_dir = "/Volumes/projects/DataSets/CSUVideo/source"
namesets = ["large_000013363_total", "large_000014631_total",
            "large_minneapolis_1_total", "large_tunisia_total"]


datum = []
for name in namesets:
    anno_prefix = "/".join([data_dir, name, "Annotations"])
    for anno_name in os.listdir(anno_prefix):
        if anno_name.startswith("."):
            continue
        anno_path = "/".join([anno_prefix, anno_name])
        # [xmin, ymin, xmax, ymax, class_id]
        labels = parse_xml(anno_path)
        if len(labels) == 0:
            continue
        datum.extend(labels)
datum = np.array(datum, np.int32)

datum_width = datum[:, 2] - datum[:, 0]
datum_height = datum[:, 3] - datum[:, 1]
datum_ratio = datum_width / datum_height

print(datum_width.shape)
print(datum_height.shape)
print(datum_ratio.shape)
d = {}
for i in datum_width:
    d.setdefault(i, 0)
    d[i] += 1
x = d.keys()
y = d.values()
plt.scatter(x, y)
plt.show()
# datum_width = datum_width.reshape((datum_width.shape[0], 1))
# datum_height = datum_height.reshape((datum_height.shape[0], 1))
# d = np.concatenate([datum_width, datum_height], axis=1)
# plt.scatter(d[:, 0], d[:, 1])
# plt.show()
# kmeans= KMeans(n_clusters=3, random_state=0).fit(datum_width)
# print(kmeans.labels_)
# print(kmeans.cluster_centers_)