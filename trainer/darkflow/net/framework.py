# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/22

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import yolov2
from os.path import basename


class base_framework(object):
    constructor = yolov2.constructor

    def __init__(self, meta, FLAGS):
        model = basename(meta['model'])
        model = '.'.join(model.split('.')[:-1])
        meta['name'] = model
        
        self.constructor(meta, FLAGS)

    def is_inp(self, file_name):
        return True

class YOLOv2(base_framework):
    constructor = yolov2.constructor
    parse = yolov2.data.parse
    shuffle = yolov2.data.shuffle
    preprocess = yolov2.predict.preprocess
    loss = yolov2.train.loss
    is_inp = yolov2.misc.is_inp
    postprocess = yolov2.predict.postprocess
    _batch = yolov2.data._batch
    resize_input = yolov2.predict.resize_input
    findboxes = yolov2.predict.findboxes
    process_box = yolov2.predict.process_box


types = {
    '[region]': YOLOv2
}

def create_framework(meta, FLAGS):
    net_type = meta['type']
    this = types.get(net_type, base_framework)
    return this(meta, FLAGS)
