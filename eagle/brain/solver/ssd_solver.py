# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from eagle.brain.solver.solver import Solver


class SSDSolver(Solver):
    def __init__(self, dataset, net, common_params, solver_params):
        super(SSDSolver, self).__init__(
            dataset, net, common_params, solver_params)

        # process params
        self.width = int(common_params['image_size'])
        self.height = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])

        self.decay = float(solver_params['decay'])
        self.beta_1 = float(solver_params['beta_1'])
        self.beta_2 = float(solver_params['beta_2'])
        self.epsilon = float(solver_params['epsilon'])
        self.learning_rate = float(solver_params['lr'])
        self.train_dir = str(solver_params['train_dir'])
        self.max_iterators = int(solver_params['max_iterators'])
        self.pretrain_path = str(solver_params['pretrain_model_path'])

        self.dataset = dataset
        self.net = net

        # construct graph
        self.build_model()

    def _train(self):
        opt = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=self.beta_1,
            beta2=self.beta_2,
            epsilon=self.epsilon)
        grads = opt.compute_gradients(self.total_loss)
        apply_gradient_op = opt.apply_gradients(grads,
                                                global_step=self.global_step)
        return apply_gradient_op

    def build_model(self):
        # construct graph
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.height, self.width, 3))
        # self.predicts = self.net.inference(self.images)
        model_spec = self.net.inference(self.images)
        self.predicts = model_spec["predictions"]
        predict_shape = model_spec["predictions"].get_shape().as_list()
        boxes_num = predict_shape[0] // self.batch_size
        encode_length = predict_shape[1]

        '''
        [32, 37, 37, 4, 8] ---> (cx, cy, w, h, variances)
        [32, 18, 18, 6, 8]
        [32,  9,  9, 6, 8]
        [32,  5,  5, 6, 8]
        [32,  3,  3, 4, 8]
        [32,  1,  1, 4, 8]
        ==> 37^2*4 + 18^2*6 + 9^2*6 + 5^2*6 + 3^2*6 + 1^2*4 = 8096
        '''

        self.labels = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, boxes_num, encode_length))

        self.total_loss = self.net.loss(y_true=self.labels,
                                        y_pred=self.predicts)

        tf.summary.scalar('loss', self.total_loss)
        self.train_op = self._train()

    def solve(self):
        saver = tf.train.Saver(max_to_keep=3)

        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()

        sess = tf.Session()
        sess.run(init)

        if os.path.isdir(self.pretrain_path):
            model_file = tf.train.latest_checkpoint(self.pretrain_path)
            saver.restore(sess, model_file)
        if os.path.isfile(self.pretrain_path):
            saver.restore(sess, self.pretrain_path)

        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

        for step in range(self.max_iterators):
            start_time = time.time()
            np_images, np_labels, np_objects_num = self.dataset.batch()

            _, loss_value, nilboy = sess.run(
                [self.train_op, self.total_loss],
                feed_dict={
                    self.images: np_images,
                    self.labels: np_labels
                })

            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = self.dataset.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.2f '
                              '(%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))
                sys.stdout.flush()
            if step % 1000 == 0:
                summary_str = sess.run(summary_op,
                                       feed_dict={
                                           self.images: np_images,
                                           self.labels: np_labels
                                       })
                summary_writer.add_summary(summary_str, step)
            if step % 5000 == 0:
                saver.save(sess,
                           self.train_dir + '/model.ckpt',
                           global_step=step)
        sess.close()
