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
from datum.models.ssd.box_encoder import BoxEncoder


class SSDDataSet(DataSet):
    """TextDataSet
    process text input file dataset
    text file format:
    image_path xmin1 ymin1 xmax1 ymax1 class1 xmin2 ymin2 xmax2 ymax2 class2
    """

    def __init__(self, common_params, dataset_params, box_encoder_params):
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

        self.upper_resize_rate = float(dataset_params["upper_resize_rate"])
        self.lower_resize_rate = float(dataset_params["lower_resize_rate"])

        self.box_encoder = BoxEncoder(common_params, box_encoder_params)

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
            if out is not None:
                # 在归整完数据之后，要对object_label中使用BoxEncoder的调用
                image, gt_labels = out[:]
                # gt_labels from
                # [xmin, ymin, xmax, ymax] --> [xmin, xmax, ymin, ymax]
                # print(gt_labels)
                for cell in gt_labels:
                    cell[1], cell[2] = cell[2], cell[1]
                # print(gt_labels)
                y_true_encoded = self.box_encoder.encode_y_sample(gt_labels)
                self.image_label_queue.put([image, y_true_encoded])

    def record_process(self, record):
        """对于每个样本的数据具体该如何处理
        Args: record --> [image_path, xmin, ymin, xmax, ymax, class_id]
        Returns:
          image: 3-D ndarray
          labels: 2-D list [[xmin, ymin, xmax, ymax, class_id]]
        """
        image = cv2.imread(record[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]

        real_rate = w / h
        target_rate = self.width / self.height

        if (target_rate - self.lower_resize_rate
                <= real_rate <= target_rate + self.upper_resize_rate):
            width_rate = self.width * 1.0 / w
            height_rate = self.height * 1.0 / h

            image = cv2.resize(image, (self.height, self.width))
            labels = []
            i = 1
            while i < len(record):
                xmin = record[i]
                ymin = record[i + 1]
                xmax = record[i + 2]
                ymax = record[i + 3]
                class_id = record[i + 4]
                labels.append([xmin * width_rate, ymin * height_rate,
                               xmax * width_rate, ymax * height_rate,
                               class_id])
                i += 5
            return [image, labels]
        elif real_rate > target_rate + self.upper_resize_rate:
            # 当前的图像不满足直接resize的比例，需要按照最短边进行一定比例进行裁减
            h0 = h
            w0 = np.ceil(h0 * (target_rate + self.upper_resize_rate)).astype(np.int32)
            # we should crop from (0, 0)
            image = image[:, 0:w0]
            image = cv2.resize(image, (self.height, self.width))
            width_rate = self.width * 1.0 / w0
            height_rate = self.height * 1.0 / h0

            # 处理原始目标区域在裁减之后的图像中的实际位置
            labels = []
            i = 1
            while i < len(record):
                xmin = record[i]
                ymin = record[i + 1]
                xmax = record[i + 2]
                ymax = record[i + 3]
                class_id = record[i + 4]
                if xmin < w0 - 1 and xmax <= w0 - 1:
                    labels.append([xmin * width_rate, ymin * height_rate,
                                   xmax * width_rate, ymax * height_rate,
                                   class_id])
                elif xmin < w0 - 1 and xmax > w0 - 1:
                    if (w0 - 1 - xmin) / (xmax - xmin) >= 0.6:
                        labels.append([xmin * width_rate, ymin * height_rate,
                                       w0-1, ymax * height_rate,
                                       class_id])
                    else:
                        pass
                else:
                    pass
                i += 5
            # 若没有目标符合变换要求，就将这个数据丢弃
            if len(labels) != 0:
                return [image, labels]
            else:
                return None
        elif real_rate < target_rate - self.lower_resize_rate:
            w0 = w
            h0 = np.ceil(w0 / (target_rate - self.lower_resize_rate)).astype(np.int32)
            # we should crop from (0, 0)
            image = image[0:h0, :]
            image = cv2.resize(image, (self.height, self.width))
            width_rate = self.width * 1.0 / w0
            height_rate = self.height * 1.0 / h0

            # 处理原始目标区域在裁减之后的图像中的实际位置
            labels = []
            i = 1
            while i < len(record):
                xmin = record[i]
                ymin = record[i + 1]
                xmax = record[i + 2]
                ymax = record[i + 3]
                class_id = record[i + 4]
                if ymin < h0 - 1 and ymax <= h0 - 1:
                    labels.append([xmin * width_rate, ymin * height_rate,
                                   xmax * width_rate, ymax * height_rate,
                                   class_id])
                elif ymin < h0 - 1 and ymax > h0 - 1:
                    if (h0 - 1 - ymin) / (ymax - ymin) >= 0.6:
                        labels.append([xmin * width_rate, ymin * height_rate,
                                       xmax * width_rate, h0 - 1,
                                       class_id])
                    else:
                        pass
                else:
                    pass
                i += 5
            # 若没有目标符合变换要求，就将这个数据丢弃
            if len(labels) != 0:
                return [image, labels]
            else:
                return None
        else:
            pass

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
          labels: (batch_size, #boxes, #classes + 4 + 4 + 4)
        """
        images = []
        labels = []
        for i in range(self.batch_size):
            image, label = self.image_label_queue.get()
            images.append(image)
            labels.append(label)
        images = np.asarray(images, dtype=np.float32)
        images = images / 255 * 2 - 1
        labels = np.concatenate(labels, axis=0)
        # labels = np.asarray(labels, dtype=np.float32)
        return images, labels
