# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
from queue import Queue
from threading import Thread

import cv2
import numpy as np

from datum.meta.BaseSet import SetMeta


class DetectDataSet(SetMeta):
    """Detect DataSet
    process text input file datum
    text file tools(each line):
        image_file_path,
        xmin1, ymin1, xmax1, ymax1, class1,
        xmin2, ymin2, xmax2, ymax2, class2
    """

    def __init__(self, common_params, dataset_params):
        super(DetectDataSet, self).__init__(common_params, dataset_params)

        self.data_path = str(dataset_params["path"])
        self.thread_num = int(dataset_params["thread_num"])

        self.width = int(common_params["image_width"])
        self.height = int(common_params["image_height"])
        self.batch_size = int(common_params["batch_size"])
        self.max_objects = int(common_params["max_objects_per_image"])

        # record and image_label queue
        self.record_queue = Queue(maxsize=10000)
        self.image_label_queue = Queue(maxsize=512)

        self.record_list = list()

        # filling the record_list
        if not os.path.isfile(self.data_path):
            raise ValueError("{} data path not found.".format(self.data_path))
        with open(self.data_path, "r") as input_file:
            for line in input_file:
                line = line.strip()
                ss = line.split(",")
                ss[1:] = [float(num) for num in ss[1:]]
                self.record_list.append(ss)
        self.record_point = 0
        self.record_number = len(self.record_list)
        self.num_batch_per_epoch = int(self.record_number / self.batch_size)

        t_record_producer = Thread(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()

        for _ in range(self.thread_num):
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
            out = self.sample_process(item)
            self.image_label_queue.put(out)

    def sample_process(self, record_sample):
        """sample process
        Parameters:
            image: 3-D ndarray
            labels: 2-D list [self.max_objects, 5] ---> (x_center, y_center,
            w, h, class_id)
            object_num: total object number
        """
        if not os.path.isfile(record_sample[0]):
            raise ValueError("image path {} invalid.".format(record_sample[0]))
        image = cv2.imread(record_sample[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        width_rate = self.width * 1.0 / w
        height_rate = self.height * 1.0 / h

        image = cv2.resize(image, (self.height, self.width))
        labels = [[0] * 5] * self.max_objects

        i, object_num = 1, 0
        while i < len(record_sample):
            xmin = record_sample[i]
            ymin = record_sample[i+1]
            xmax = record_sample[i+2]
            ymax = record_sample[i+3]
            class_id = record_sample[i+4]

            x_center = (xmin + xmax) * 1.0 / 2 * width_rate
            y_center = (ymin + ymax) * 1.0 / 2 * height_rate

            box_w = (xmax - xmin) * width_rate
            box_h = (ymax - ymin) * height_rate

            labels[object_num] = [x_center, y_center, box_w, box_h, class_id]

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
        images, labels, objects_num = list(), list(), list()
        for _ in range(self.batch_size):
            image, label, object_num = self.image_label_queue.get()
            images.append(image)
            labels.append(label)
            objects_num.append(object_num)

        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        objects_num = np.asarray(objects_num, dtype=np.int32)
        return images, labels, objects_num
