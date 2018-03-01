# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/27

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random
import threading
import numpy as np
import multiprocessing

if sys.version_info[0] == 2:
    import cPickle as pickle
    from Queue import Empty as QueueEmpty
elif sys.version_info[0] == 3:
    import pickle
    from queue import Empty as QueueEmpty

import eagle.utils as eu


class Batch(object):
    """
    Class encapsulating a batch before and after augmentation.

    Parameters
    ----------
    images : None or (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
        The images to
        augment.

    keypoints : None or list of KeypointOnImage
        The keypoints to
        augment.

    data : anything
        Additional data that is saved in the batch and may be read out
        after augmentation. This could e.g. contain filepaths to each image
        in `images`. As this object is usually used for background
        augmentation with multiple processes, the augmented Batch objects might
        not be returned in the original order, making this information useful.
    """
    def __init__(self, images=None, keypoints=None, data=None):
        self.images = images
        self.images_aug = None
        self.keypoints = keypoints
        self.keypoints_aug = None
        self.data = data


class BatchLoader(object):
    """
    Class to load batches in the background.

    Loaded batches can be accesses using `BatchLoader.queue`.

    Parameters
    ----------
    load_batch_func : callable
        Function that yields Batch objects (i.e. expected to be a generator).
        Background loading automatically stops when the last batch was yielded.

    queue_size : int, optional(default=50)
        Maximum number of batches to store in the queue. May be set higher
        for small images and/or small batches.

    nb_workers : int, optional(default=1)
        Number of workers to run in the background.

    threaded : bool, optional(default=True)
        Whether to run the background processes using threads (true) or
        full processes (false).
    """
    def __init__(self,
                 load_batch_func, queue_size=50, nb_workers=1, threaded=True):
        eu.do_assert(queue_size > 0)
        eu.do_assert(nb_workers >= 1)
        self.queue = multiprocessing.Queue(queue_size)
        self.join_signal = multiprocessing.Event()
        self.finished_signals = []
        self.workers = []
        self.threaded = threaded
        seeds = eu.current_random_state().randint(
            0, 10 ** 6, size=(nb_workers,))
        for i in range(nb_workers):
            finished_signal = multiprocessing.Event()
            self.finished_signals.append(finished_signal)
            if threaded:
                worker = threading.Thread(
                    target=self._load_batches,
                    args=(
                        load_batch_func,
                        self.queue,
                        finished_signal,
                        self.join_signal,
                        None))
            else:
                worker = multiprocessing.Process(
                    target=self._load_batches,
                    args=(load_batch_func,
                          self.queue,
                          finished_signal,
                          self.join_signal,
                          seeds[i]))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def all_finished(self):
        """
        Determine whether the workers have finished the loading process.
        Returns
        -------
        out : bool
            True if all workers have finished. Else False.
        """
        return all([event.is_set() for event in self.finished_signals])

    def _load_batches(self, load_batch_func,
                      queue, finished_signal, join_signal, seedval):
        if seedval is not None:
            random.seed(seedval)
            np.random.seed(seedval)
            eu.seed(seedval)

        for batch in load_batch_func():
            eu.do_assert(isinstance(batch, Batch),
                         message="Expected batch returned by lambda "
                                 "function to be of class Batch, got " +
                                 str(type(Batch)))
            queue.put(pickle.dumps(batch, protocol=-1))
            if join_signal.is_set():
                break

        finished_signal.set()

    def terminate(self):
        """Stop all workers."""
        self.join_signal.set()
        if self.threaded:
            for worker in self.workers:
                worker.join()
        else:
            for worker, finished_signal in zip(self.workers, self.finished_signals):
                worker.terminate()
                finished_signal.set()


class BackgroundAugmentor(object):
    """
    Class to augment batches in the background (while training on the GPU).

    This is a wrapper around the multiprocessing module.

    Parameters
    ----------
    batch_loader : BatchLoader
        BatchLoader object to load data in the
        background.

    augseq : Augmenter
        An augmenter to apply to all loaded images.
        This may be e.g. a Sequential to apply multiple augmenters.

    queue_size : int
        Size of the queue that is used to temporarily save the augmentation
        results. Larger values offer the background processes more room
        to save results when the main process doesn't load much, i.e. they
        can lead to smoother and faster training. For large images, high
        values can block a lot of RAM though.

    nb_workers : "auto" or int
        Number of background workers to spawn. If auto, it will be set
        to C-1, where C is the number of CPU cores.
    """
    def __init__(self, batch_loader, augseq, queue_size=50, nb_workers="auto"):
        eu.do_assert(queue_size > 0)
        self.augseq = augseq
        self.source_finished_signals = batch_loader.finished_signals
        self.queue_source = batch_loader.queue
        self.queue_result = multiprocessing.Queue(queue_size)

        if nb_workers == "auto":
            try:
                nb_workers = multiprocessing.cpu_count()
            except (ImportError, NotImplemented):
                nb_workers = 1
            nb_workers = max(1, nb_workers - 1)
        else:
            eu.do_assert(nb_workers >= 1)
        print("Starting {} background processes ...".format(nb_workers))

        self.nb_workers = nb_workers
        self.workers = []
        self.nb_workers_finished = 0
        self.augment_images = True
        self.augment_keypoints = True

        seeds = eu.current_random_state().randint(0, 10**6, size=(nb_workers,))
        for i in range(nb_workers):
            worker = multiprocessing.Process(
                target=self._augment_images_worker,
                args=(augseq, self.queue_source, self.queue_result,
                      self.source_finished_signals, seeds[i]))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def _augment_images_worker(self, augseq,
                               queue_source, queue_result,
                               source_finished_signals, seedval):
        np.random.seed(seedval)
        random.seed(seedval)
        augseq.reseed(seedval)
        eu.seed(seedval)

        while True:
            """wait for a new batch in the source queue and load it."""
            try:
                batch_str = queue_source.get(timeout=0.1)
                batch = pickle.loads(batch_str)
                batch_augment_images = batch.images is not None and \
                                       self.augment_images
                batch_augment_kps = batch.keypoints is not None and \
                                    self.augment_keypoints
                if batch_augment_images and batch_augment_kps:
                    augseq_det = augseq.to_deterministic() \
                        if not augseq.deterministic else augseq
                    batch.images_aug = augseq_det.augment_images(batch.images)
                    batch.keypoints_aug = augseq_det.augment_keypoints(batch.keypoints)
                elif batch_augment_images:
                    batch.images_aug = augseq.augment_images(batch.images)
                elif batch_augment_kps:
                    batch.keypoints_aug = augseq.augment_keypoints(
                        batch.keypoints)
                """send augmented batch to ouput queue"""
                batch_str = pickle.dumps(batch, protocol=-1)
                queue_result.put(batch_str)
            except QueueEmpty:
                if all([signal.is_set() for signal in source_finished_signals]):
                    queue_result.put(pickle.dumps(None, protocol=-1))
                    return

    def get_batch(self):
        """
        Return a batch from the queue of augment batches.
        If workers are still running and there are no batches in the queue,
        it will automatically wait for the next batch.
        """
        batch_str = self.queue_result.get()
        batch = pickle.loads(batch_str)
        if batch is not None:
            return batch
        else:
            self.nb_workers_finished += 1
            if self.nb_workers_finished == self.nb_workers:
                return None
            else:
                return self.get_batch()

    def terminate(self):
        """Terminates all background processes immediately.
        This will alsofree their RAM."""
        for worker in self.workers:
            worker.terminate()


class HooksImages(object):
    """
    Class to intervene with image augmentation runs.

    This is e.g. useful to dynamically deactivate some augmenters.

    Parameters
    ----------
    activator : None or callable, optional(default=None)
        A function that gives permission to execute an augmenter.
        The expected interface is
            `f(images, augmenter, parents, default)`,
        where `images` are the input images to augment, `augmenter` is the
        instance of the augmenter to execute, `parents` are previously
        executed augmenters and `default` is an expected default value to be
        returned if the activator function does not plan to make a decision
        for the given inputs.

    propagator : None or callable, optional(default=None)
        A function that gives permission to propagate the augmentation further
        to the children of an augmenter. This happens after the activator.
        In theory, an augmenter may augment images itself (if allowed by the
        activator) and then execute child augmenters afterwards (if allowed by
        the propagator). If the activator returned False, the propagation step
        will never be executed.
        The expected interface is
            `f(images, augmenter, parents, default)`,
        with all arguments having identical meaning to the activator.

    preprocessor : None or callable, optional(default=None)
        A function to call before an augmenter performed any augmentations.
        The interface is
            `f(images, augmenter, parents)`,
        with all arguments having identical meaning to the activator.
        It is expected to return the input images, optionally modified.

    postprocessor : None or callable, optional(default=None)
        A function to call after an augmenter performed augmentations.
        The interface is the same as for the preprocessor.

    Examples
    --------
    >>> seq = iaa.Sequential([
    >>>     iaa.GaussianBlur(3.0, name="blur"),
    >>>     iaa.Dropout(0.05, name="dropout"),
    >>>     iaa.Affine(translate_px=-5, name="affine")
    >>> ])
    >>>
    >>> def activator(images, augmenter, parents, default):
    >>>     return False if augmenter.name in ["blur", "dropout"] else default
    >>>
    >>> seq_det = seq.to_deterministic()
    >>> images_aug = seq_det.augment_images(images)
    >>> heatmaps_aug = seq_det.augment_images(
    >>>     heatmaps,
    >>>     hooks=ia.HooksImages(activator=activator)
    >>> )

    This augments images and their respective heatmaps in the same way.
    The heatmaps however are only modified by Affine, not by GaussianBlur or
    Dropout.

    """

    def __init__(self, activator=None, propagator=None, preprocessor=None, postprocessor=None):
        self.activator = activator
        self.propagator = propagator
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

    def is_activated(self, images, augmentor, parents, default):
        """
        Returns whether an augmenter may be executed.

        Returns
        -------
        out : bool
            If True, the augmenter may be executed. If False, it may
            not be executed.

        """
        if self.activator is None:
            return default
        else:
            return self.activator(images, augmentor, parents, default)

    # TODO is a propagating hook necessary? seems to be covered by activated
    # hook already
    def is_propagating(self, images, augmentor, parents, default):
        """
        Returns whether an augmenter may call its children to augment an
        image. This is independent of the augmenter itself possible changing
        the image, without calling its children. (Most (all?) augmenters with
        children currently dont perform any changes themselves.)

        Returns
        -------
        out : bool
            If True, the augmenter may be propagate to its children.
            If False, it may not.

        """
        if self.propagator is None:
            return default
        else:
            return self.propagator(images, augmentor, parents, default)

    def preprocess(self, images, augmentor, parents):
        """
        A function to be called before the augmentation of images starts (per
        augmenter).

        Returns
        -------
        out : (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            The input images, optionally modified.

        """
        if self.preprocessor is None:
            return images
        else:
            return self.preprocessor(images, augmentor, parents)

    def postprocess(self, images, augmentor, parents):
        """
        A function to be called after the augmentation of images was
        performed.

        Returns
        -------
        out : (N,H,W,C) ndarray or (N,H,W) ndarray or list of (H,W,C) ndarray or list of (H,W) ndarray
            The input images, optionally modified.

        """
        if self.postprocessor is None:
            return images
        else:
            return self.postprocessor(images, augmentor, parents)


class HooksKeyPoints(HooksImages):
    """
    Class to intervene with keypoint augmentation runs.
    This is e.g. useful to dynamically deactivate some augments.
    This class is currently the same as the one for images. This may or may
    not change in the futures.
    """
    pass
