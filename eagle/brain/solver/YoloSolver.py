# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/5

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os, sys
import numpy as np
import tensorflow as tf
from datetime import datetime

from eagle.brain.solver.BaseSolver import BaseSolver


class YoloSolver(BaseSolver):

    def __init__(self, dataset, net, common_params, solver_params):
        self.width = int(common_params["image_width"])
        self.height = int(common_params["image_height"])
        self.batch_size = int(common_params["batch_size"])
        self.max_objects = int(common_params["max_objects_per_image"])

        self.lr = float(solver_params["lr"])
        self.moment = float(solver_params["moment"])
        self.train_dir = str(solver_params["train_dir"])
        self.max_iterators = int(solver_params["max_iterators"])
        self.pretrain_path = str(solver_params["pretrain_model_path"])

        if self.train_dir[-1] != '/':
            self.train_dir += "/"
        # if self.pretrain_path == "None":
        #     self.pretrain_path = None

        self.dataset = dataset
        self.net = net

        self.build_model()

    def build_model(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.images = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.height, self.width, 3))
        self.labels = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.max_objects, 5))
        self.objects_num = tf.placeholder(tf.int32, (self.batch_size))

        self.predicts = self.net.inference(self.images)
        self.total_loss, self.nilboy = self.net.loss(self.predicts,
                                                     self.labels,
                                                     self.objects_num)

        tf.summary.scalar("loss", self.total_loss)
        self.train_op = self.create_train_op()

    def create_train_op(self):
        """Train model
        Create an optimizer and apply to all trainable variables.
        Parameters:
            total_loss: Total loss from net.loss()
            global_step: counting the number of training steps processed
        Returns:
            train_op: op for training
        """
        opt = tf.train.MomentumOptimizer(self.lr, self.moment)
        grads = opt.compute_gradients(self.total_loss)
        apply_gradient_op = opt.apply_gradients(grads,
                                                global_step=self.global_step)
        return apply_gradient_op

    def solve(self):
        saver_pretrain = tf.train.Saver(self.net.pretrained_collection)
        saver_train = tf.train.Saver(self.net.trainable_collection)

        init = tf.global_variables_initializer()
        summary_op = tf.summary.merge_all()

        sess = tf.Session()
        sess.run(init)

        if os.path.isdir(self.pretrain_path):
            saver_pretrain.restore(sess, self.pretrain_path)
        if not os.path.isdir(self.train_dir):
            os.makedirs(self.train_dir)
        summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

        for step in range(self.max_iterators):
            start_time = time.time()
            np_images, np_labels, np_objects_num = self.dataset.batch()

            _, loss_val, nilboy = sess.run(
                [self.train_op, self.total_loss, self.nilboy],
                feed_dict={
                    self.images: np_images,
                    self.labels: np_labels,
                    self.objects_num: np_objects_num
                })

            duration = time.time() - start_time

            assert not np.isnan(loss_val), "Model diverged with loss=NaN"

            if step % 1 == 0:
                num_examples_per_step = self.dataset.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                format_str = ("{}: step {}, loss = {} "
                              "({} examples/sec; {} sec/batch)")
                print(format_str.format(datetime.now(), step, loss_val,
                                        examples_per_sec, sec_per_batch))

                sys.stdout.flush()

            if step % 1 == 0:
                summary_str = sess.run(
                    summary_op,
                    feed_dict={
                        self.images: np_images,
                        self.labels: np_labels,
                        self.objects_num: np_objects_num
                    })
                summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0:
                if not os.path.isdir(self.train_dir):
                    os.makedirs(self.train_dir)
                saver_train.save(sess,
                                 self.train_dir + "model.ckpt",
                                 global_step=step)
        sess.close()
