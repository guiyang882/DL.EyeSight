# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/7

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
from queue import Queue
from threading import Thread

import cv2
import numpy as np

from datum.meta.dataset import DataSet


class SSDDataSet(DataSet):
    """TextDataSet
    process text input file dataset
    text file format:
    image_path xmin1 ymin1 xmax1 ymax1 class1 xmin2 ymin2 xmax2 ymax2 class2
    """

    def __init__(self, common_params, dataset_params):
        super(SSDDataSet, self).__init__(common_params, dataset_params)

        # process params
        self.width = int(common_params['image_size'])
        self.height = int(common_params['image_size'])
        self.channel = int(common_params["image_channel"])
        self.batch_size = int(common_params['batch_size'])
        self.num_classes = int(common_params["num_classes"])

        self.data_path = str(dataset_params['path'])
        self.thread_num = int(dataset_params['thread_num'])
        self.classes = json.loads(dataset_params["classes"])
        self.box_output_format = json.loads(dataset_params["box_output_format"])
        self.is_need_bg = True if dataset_params["is_need_bg"] == "True" else False

        # record and image_label queue
        self.record_queue = Queue(maxsize=10000)
        self.image_label_queue = Queue(maxsize=2000)

        self.record_list = []

        # filling the record_list
        input_file = open(self.data_path, 'r')

        for line in input_file:
            line = line.strip()
            ss = line.split(' ')
            ss[1:] = [float(num) for num in ss[1:]]
            # 文件中存储的类别都是从0开始的，如果需要在处理前添加background这个类别
            # 需要将background这个设置为0，其他的类别编号自动+1
            if self.is_need_bg:
                self.classes.insert(0, "background")
                step_len = len(self.box_output_format)
                start_class_idx = self.box_output_format.index("class_id") + 1
                for i in range(start_class_idx, len(ss), step_len):
                    ss[i] += 1
            self.record_list.append(ss)

        self.record_point = 0
        self.record_number = len(self.record_list)

        self.num_batch_per_epoch = int(self.record_number / self.batch_size)

        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()

        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start()

    def record_producer(self):
        while True:
            if self.record_point % self.record_number == 0:
                random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point += 1

    def record_customer(self):
        while True:
            item = self.record_queue.get()
            out = self.record_process(item)
            self.image_label_queue.put(out)

    def record_process(self, record):
        """对于每个样本的数据具体该如何处理
        Args: record --> [image_path, xmin, ymin, xmax, ymax, class_id]
        Returns:
          image: 3-D ndarray
          labels: 2-D list [self.max_objects, 5]
                ---> (xcenter, ycenter, w, h, objects_num)
          object_num:  total object number  int
        """
        image = cv2.imread(record[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]

        width_rate = self.width * 1.0 / w
        height_rate = self.height * 1.0 / h

        image = cv2.resize(image, (self.height, self.width))

        labels = [[0, 0, 0, 0, 0]] * self.max_objects
        i = 1
        object_num = 0
        while i < len(record):
            xmin = record[i]
            ymin = record[i + 1]
            xmax = record[i + 2]
            ymax = record[i + 3]
            class_num = record[i + 4]

            xcenter = (xmin + xmax) * 1.0 / 2 * width_rate
            ycenter = (ymin + ymax) * 1.0 / 2 * height_rate

            box_w = (xmax - xmin) * width_rate
            box_h = (ymax - ymin) * height_rate

            labels[object_num] = [xcenter, ycenter, box_w, box_h, class_num]
            object_num += 1
            i += 5
            if object_num >= self.max_objects:
                break
        return [image, labels, object_num]

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
          labels: 3-D ndarray [batch_size, max_objects, 5]
          objects_num: 1-D ndarray [batch_size]
        """
        images = []
        labels = []
        objects_num = []
        for i in range(self.batch_size):
            image, label, object_num = self.image_label_queue.get()
            images.append(image)
            labels.append(label)
            objects_num.append(object_num)
        images = np.asarray(images, dtype=np.float32)
        images = images / 255 * 2 - 1
        labels = np.asarray(labels, dtype=np.float32)
        objects_num = np.asarray(objects_num, dtype=np.int32)
        return images, labels, objects_num