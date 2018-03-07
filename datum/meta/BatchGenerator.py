# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2017/12/18

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
Includes:
 * A batch gnerator for SSD model training and inference which can perform online data agumentation
 * An offline image processor that saves processed images and adjusted labels to disk
"""

import os
import pickle
import csv
import random
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image
import sklearn.utils
from bs4 import BeautifulSoup


"""
Image Processing functions used by the generator to perform the following image manipulations:
 * Translation
 * Horizontal flip
 * Scaling
 * Brightness change
 * Histogram contrast equalization
"""

class BatchGenerator:
    def __init__(self,
                 box_output_format=['class_id', 'xmin', 'xmax', 'ymin', 'ymax'],
                 filenames=None,
                 filenames_type='text',
                 images_path=None,
                 labels=None):
        '''
        This class provides parser methods that you call separately after calling the constructor to assemble
        the list of image filenames and the list of labels for the datum from CSV or XML files. If you already
        have the image filenames and labels in asuitable tools (see argument descriptions below), you can pass
        them right here in the constructor, in which case you do not need to call any of the parser methods afterwards.
        In case you would like not to load any labels at all, simply pass a list of image filenames here.
        Arguments:
            box_output_format (list, optional): A list of five strings representing the desired order of the five
                items class ID, xmin, xmax, ymin, ymax in the generated data. The expected strings are
                'xmin', 'xmax', 'ymin', 'ymax', 'class_id'. If you want to train the model, this
                must be the order that the box encoding class requires as input. Defaults to
                `['class_id', 'xmin', 'xmax', 'ymin', 'ymax']`. Note that even though the parser methods are
                able to produce different output formats, the SSDBoxEncoder currently requires the tools
                `['class_id', 'xmin', 'xmax', 'ymin', 'ymax']`. This list only specifies the five box parameters
                that are relevant as training targets, a list of filenames is generated separately.
            filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
                a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. Note that the list/tuple must contain the paths to the images,
                not the images themselves. If a filepath string is passed, it must point either to
                (1) a pickled file containing a list/tuple as described above. In this case the `filenames_type`
                argument must be set to `pickle`.
                Or
                (2) a text file. Each line of the text file contains the file name (basename of the file only,
                not the full directory path) to one image and nothing else. In this case the `filenames_type`
                argument must be set to `text` and you must pass the path to the directory that contains the
                images in `images_path`.
            filenames_type (string, optional): In case a string is passed for `filenames`, this indicates what
                type of file `filenames` is. It can be either 'pickle' for a pickled file or 'text' for a
                plain text file. Defaults to 'text'.
            images_path (string, optional): In case a text file is passed for `filenames`, the full paths to
                the images will be composed from `images_path` and the names in the text file, i.e. this
                should be the directory that contains the images to which the text file refers.
                If `filenames_type` is not 'text', then this argument is irrelevant. Defaults to `None`.
            labels (string or list, optional): `None` or either a Python list/tuple or a string representing
                the path to a pickled file containing a list/tuple. The list/tuple must contain Numpy arrays
                that represent the labels of the datum.
        '''
        # These are the variables we always need
        self.include_classes = None
        self.box_output_format = box_output_format

        # These are the variables that we only need if we want to use parse_csv()
        self.images_path = None
        self.labels_path = None
        self.input_format = None

        # These are the variables that we only need if we want to use parse_xml()
        self.images_paths = None
        self.annotations_path = None
        self.image_set_path = None
        self.image_set = None
        self.classes = None

        # The two variables below store the output from the parsers. This is the input for the generate() method.
        # `self.filenames` is a list containing all file names of the image samples (full paths). Note that it does not contain the actual image files themselves.
        # `self.labels` is a list containing one 2D Numpy array per image. For an image with `k` ground truth bounding boxes,
        # the respective 2D array has `k` rows, each row containing `(xmin, xmax, ymin, ymax, class_id)` for the respective bounding box.
        # Setting `self.labels` is optional, the generator also works if `self.labels` remains `None`.

        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    if filenames_type == 'pickle':
                        self.filenames = pickle.load(f)
                    elif filenames_type == 'text':
                        self.filenames = [os.path.join(images_path, line.strip()) for line in f]
                    else:
                        raise ValueError("`filenames_type` can be either 'text' or 'pickle'.")
            else:
                raise ValueError("`filenames` must be either a Python list/tuple or a string representing a filepath (to a pickled or text file). The value you passed is neither of the two.")
        else:
            self.filenames = [] # All unique image filenames will go here.

        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError("`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None # This will be either `None` or a list of 2D Numpy arrays with all the ground truth boxes for a given image.

    def parse_xml(self,
                  images_paths=None,
                  annotations_paths=None,
                  image_set_paths=None,
                  classes=['background',
                           'aeroplane', 'bicycle', 'bird', 'boat',
                           'bottle', 'bus', 'car', 'cat',
                           'chair', 'cow', 'diningtable', 'dog',
                           'horse', 'motorbike', 'person', 'pottedplant',
                           'sheep', 'sofa', 'train', 'tvmonitor'],
                  include_classes = 'all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False):
        '''
        This is an XML parser for the Pascal VOC datasets. It might be applicable to other datasets with minor changes to
        the code, but in its current form it expects the data tools and XML tags of the Pascal VOC datasets.
        Arguments:
            images_paths (str, optional):
            annotations_paths (str, optional): The path to the directory that contains the annotation XML files for
                the images. The directory must contain one XML file per image and name of the XML file must be the
                image ID. The content of the XML files must be in the Pascal VOC tools. Defaults to `None`.
            image_set_paths (str, optional): The path to the text file with the image
                set to be loaded. This text file simply contains one image ID per line and nothing else. Defaults to `None`.
            classes (list, optional): A list containing the names of the object classes as found in the
                `name` XML tags. Must include the class `background` as the first list item. The order of this list
                defines the class IDs. Defaults to the list of Pascal VOC classes in alphabetical order.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the datum. Defaults to 'all', in which case all boxes will be included
                in the datum.
            exclude_truncated (bool, optional): If `True`, excludes boxes that are labeled as 'truncated'.
                Defaults to `False`.
            exclude_difficult (bool, optional): If `True`, excludes boxes that are labeled as 'difficult'.
                Defaults to `False`.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.
                Defaults to `False`.
        Returns:
            None by default, optionally the image filenames and labels.
        '''

        if not images_paths is None: self.images_paths = images_paths
        if not annotations_paths is None: self.annotations_paths = annotations_paths
        if not image_set_paths is None: self.image_set_paths = image_set_paths
        if not classes is None: self.classes = classes
        if not include_classes is None: self.include_classes = include_classes

        # Erase data that might have been parsed before
        self.filenames = []
        self.labels = []

        for image_path, image_set_path, annotations_path in zip(self.images_paths, self.image_set_paths, self.annotations_paths):
            # Parse the image set that so that we know all the IDs of all the images to be included in the datasource
            with open(image_set_path) as f:
                image_ids = [line.strip() for line in f]

            # Parse the labels for each image ID from its respective XML file
            for image_id in image_ids:
                # Open the XML file for this image
                with open(os.path.join(annotations_path, image_id+'.xml')) as f:
                    soup = BeautifulSoup(f, 'xml')

                folder = soup.folder.text # In case we want to return the folder in addition to the image file name. Relevant for determining which datasource an image belongs to.
                filename = soup.filename.text
                self.filenames.append(os.path.join(image_path, filename))

                boxes = [] # We'll store all boxes for this image here
                objects = soup.find_all('object') # Get a list of all objects in this image

                # Parse the data for each object
                for obj in objects:
                    class_name = obj.find('name').text
                    class_id = self.classes.index(class_name)
                    # Check if this class is supposed to be included in the datasource
                    if (not self.include_classes == 'all') and (not class_id in self.include_classes): continue
                    # pose = obj.pose.text
                    pose = 'Unknown'
                    # truncated = int(obj.truncated.text)
                    truncated = 0
                    if exclude_truncated and (truncated ==1): continue
                    difficult = int(obj.difficult.text)
                    if exclude_difficult and (difficult == 1): continue
                    xmin = int(obj.bndbox.xmin.text)
                    ymin = int(obj.bndbox.ymin.text)
                    xmax = int(obj.bndbox.xmax.text)
                    ymax = int(obj.bndbox.ymax.text)
                    item_dict = {'folder': folder,
                                 'image_name': filename,
                                 'image_id': image_id,
                                 'class_name': class_name,
                                 'class_id': class_id,
                                 'pose': pose,
                                 'truncated': truncated,
                                 'difficult': difficult,
                                 'xmin': xmin,
                                 'ymin': ymin,
                                 'xmax': xmax,
                                 'ymax': ymax}
                    box = []
                    for item in self.box_output_format:
                        box.append(item_dict[item])
                    boxes.append(box)

                self.labels.append(boxes)

        if ret:
            return self.filenames, self.labels

    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 train=True,
                 ssd_box_encoder=None,
                 equalize=False,
                 brightness=False,
                 flip=False,
                 translate=False,
                 scale=False,
                 max_crop_and_resize=False,
                 full_crop_and_resize=False,
                 random_crop=False,
                 crop=False,
                 resize=False,
                 gray=False,
                 limit_boxes=True,
                 include_thresh=0.3,
                 diagnostics=False):
        '''
        Generate batches of samples and corresponding labels indefinitely from
        lists of filenames and labels.
        Returns two Numpy arrays, one containing the next `batch_size` samples
        from `filenames`, the other containing the corresponding labels from
        `labels`.
        Can shuffle `filenames` and `labels` consistently after each complete pass.
        Can perform image transformations for data conversion and data augmentation.
        `resize`, `gray`, and `equalize` are image conversion tools and should be
        used consistently during training and inference. The remaining transformations
        serve for data augmentation. Each data augmentation process can set its own
        independent application probability. The transformations are performed
        in the order of their arguments, i.e. equalization is performed first,
        grayscale conversion is performed last.
        `prob` works the same way in all arguments in which it appears. It must be a float in [0,1]
        and determines the probability that the respective transform is applied to a given image.
        All conversions and transforms default to `False`.
        Arguments:
            batch_size (int, optional): The size of the batches to be generated. Defaults to 32.
            shuffle (bool, optional): Whether or not to shuffle the datum before each pass. Defaults to `True`.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            train (bool, optional): Whether or not the generator is used in training mode. If `True`, then the labels
                will be transformed into the tools that the SSD cost function requires. Otherwise,
                the output tools of the labels is identical to the input tools. Defaults to `True`.
            ssd_box_encoder (SSDBoxEncoder, optional): Only required if `train = True`. An SSDBoxEncoder object
                to encode the ground truth labels to the required tools for training an SSD model.
            equalize (bool, optional): If `True`, performs histogram equalization on the images.
                This can improve contrast and lead the improved model performance.
            brightness (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the brightness of the image by a factor randomly picked from a uniform
                distribution in the boundaries of `[min, max]`. Both min and max must be >=0.
            flip (float, optional): `False` or a float in [0,1], see `prob` above. Flip the image horizontally.
                The respective box coordinates are adjusted accordingly.
            translate (tuple, optional): `False` or a tuple, with the first two elements tuples containing
                two integers each, and the third element a float: `((min, max), (min, max), prob)`.
                The first tuple provides the range in pixels for the horizontal shift of the image,
                the second tuple for the vertical shift. The number of pixels to shift the image
                by is uniformly distributed within the boundaries of `[min, max]`, i.e. `min` is the number
                of pixels by which the image is translated at least. Both `min` and `max` must be >=0.
                The respective box coordinates are adjusted accordingly.
            scale (tuple, optional): `False` or a tuple containing three floats, `(min, max, prob)`.
                Scales the image by a factor randomly picked from a uniform distribution in the boundaries
                of `[min, max]`. Both min and max must be >=0.
            max_crop_and_resize (tuple, optional): `False` or a tuple of four integers, `(height, width, min_1_object, max_#_trials)`.
                This will crop out the maximal possible image patch with an aspect ratio defined by `height` and `width` from the
                input image and then resize the resulting patch to `(height, width)`. The latter two components of the tuple work
                identically as in `random_crop`. Note the difference to `random_crop`: This operation crops patches of variable size
                and fixed aspect ratio from the input image and then resizes the patch, while `random_crop` crops patches of fixed
                size and fixed aspect ratio from the input image. If this operation is active, it overrides both
                `random_crop` and `resize`.
            full_crop_and_resize (tuple, optional): `False` or a tuple of four integers and one float,
                `(height, width, min_1_object, max_#_trials, mix_ratio)`. This will generate a patch of size `(height, width)`
                that always contains the full input image. The latter third and fourth components of the tuple work identically as
                in `random_crop`. `mix_ratio` is only relevant if `max_crop_and_resize` is active, in which case it must be a float in
                `[0, 1]` that decides what ratio of images will be processed using `max_crop_and_resize` and what ratio of images
                will be processed using `full_crop_and_resize`. If `mix_ratio` is 1, all images will be processed using `full_crop_and_resize`.
                Note the difference to `max_crop_and_resize`: While `max_crop_and_resize` will crop out the largest possible patch
                that still lies fully within the input image, the patch generated here will always contain the full input image.
                If this operation is active, it overrides both `random_crop` and `resize`.
            random_crop (tuple, optional): `False` or a tuple of four integers, `(height, width, min_1_object, max_#_trials)`,
                where `height` and `width` are the height and width of the patch that is to be cropped out at a random
                position in the input image. Note that `height` and `width` can be arbitrary - they are allowed to be larger
                than the image height and width, in which case the original image will be randomly placed on a black background
                canvas of size `(height, width)`. `min_1_object` is either 0 or 1. If 1, there must be at least one detectable
                object remaining in the image for the crop to be valid, and if 0, crops with no detectable objects left in the
                image patch are allowed. `max_#_trials` is only relevant if `min_1_object == 1` and sets the maximum number
                of attempts to get a valid crop. If no valid crop was obtained within this maximum number of attempts,
                the respective image will be removed from the batch without replacement (i.e. for each removed image, the batch
                will be one sample smaller). Defaults to `False`.
            crop (tuple, optional): `False` or a tuple of four integers, `(crop_top, crop_bottom, crop_left, crop_right)`,
                with the number of pixels to crop off of each side of the images.
                The targets are adjusted accordingly. Note: Cropping happens before resizing.
            resize (tuple, optional): `False` or a tuple of 2 integers for the desired output
                size of the images in pixels. The expected tools is `(height, width)`.
                The box coordinates are adjusted accordingly. Note: Resizing happens after cropping.
            gray (bool, optional): If `True`, converts the images to grayscale. Note that the resulting grayscale
                images have shape `(height, width, 1)`.
            limit_boxes (bool, optional): If `True`, limits box coordinates to stay within image boundaries
                post any transformation. This should always be set to `True`, even if you set `include_thresh`
                to 0. I don't even know why I made this an option. If this is set to `False`, you could
                end up with some boxes that lie entirely outside the image boundaries after a given transformation
                and such boxes would of course not make any sense and have a strongly adverse effect on the learning.
            include_thresh (float, optional): Only relevant if `limit_boxes` is `True`. Determines the minimum
                fraction of the area of a ground truth box that must be left after limiting in order for the box
                to still be included in the batch data. If set to 0, all boxes are kept except those which lie
                entirely outside of the image bounderies after limiting. If set to 1, only boxes that did not
                need to be limited at all are kept. Defaults to 0.3.
            diagnostics (bool, optional): If `True`, yields three additional output items:
                1) A list of the image file names in the batch.
                2) An array with the original, unaltered images.
                3) A list with the original, unaltered labels.
                This can be useful for diagnostic purposes. Defaults to `False`. Only works if `train = True`.
        Yields:
            The next batch as either of
            (1) a 3-tuple containing a Numpy array that contains the images, a Python list
            that contains the corresponding labels for each image as 2D Numpy arrays, and another Python list
            that contains the file names of the images in the batch. This is the case if `train==False`
            and labels are available.
            (2) a 2-tuple containing a Numpy array that contains the images and a Python list
            that contains the file names of the images in the batch. This is the case if `train==False`
            and labels are not available.
            (3) a 2-tuple containing a Numpy array that contains the images and another Numpy array with the
            labels in the tools that `SSDBoxEncoder.encode_y()` returns, namely an array with shape
            `(batch_size, #boxes, #classes + 4 + 4 + 4)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 4 ground truth coordinate offsets, 4 anchor box coordinates, 4 variances]`.
            The tools and order of the box coordinates is according to the `box_output_format` that was specified
            in the `BachtGenerator` constructor.
        '''

        if shuffle: # Shuffle the data before we begin
            if self.labels is None:
                self.filenames = sklearn.utils.shuffle(self.filenames)
            else:
                self.filenames, self.labels = sklearn.utils.shuffle(self.filenames, self.labels)
        current = 0

        # Find out the indices of the box coordinates in the label data
        xmin = self.box_output_format.index('xmin')
        xmax = self.box_output_format.index('xmax')
        ymin = self.box_output_format.index('ymin')
        ymax = self.box_output_format.index('ymax')

        while True:

            batch_X, batch_y = [], []

            if current >= len(self.filenames):
                current = 0
                if shuffle: # Shuffle the data after each complete pass
                    if self.labels is None:
                        self.filenames = sklearn.utils.shuffle(self.filenames)
                    else:
                        self.filenames, self.labels = sklearn.utils.shuffle(self.filenames, self.labels)

            for filename in self.filenames[current:current+batch_size]:
                with Image.open(filename) as img:
                    batch_X.append(np.array(img))

            if not self.labels is None:
                batch_y = deepcopy(self.labels[current:current+batch_size])
            else:
                batch_y = None

            this_filenames = self.filenames[current:current+batch_size] # The filenames of the files in the current batch

            current += batch_size

            for i in range(len(batch_X)):

                img_height, img_width, ch = batch_X[i].shape
                if not batch_y is None:
                    batch_y[i] = np.array(batch_y[i]) # Convert labels into an array (in case it isn't one already), otherwise the indexing below breaks

                if max_crop_and_resize:
                    # The ratio of the two aspect ratios (source image and target size) determines the maximal possible crop.
                    image_aspect_ratio = img_width / img_height
                    resize_aspect_ratio = max_crop_and_resize[1] / max_crop_and_resize[0]

                    if image_aspect_ratio < resize_aspect_ratio:
                        crop_width = img_width
                        crop_height = int(round(crop_width / resize_aspect_ratio))
                    else:
                        crop_height = img_height
                        crop_width = int(round(crop_height * resize_aspect_ratio))
                    # The actual cropping and resizing will be done by the random crop and resizing operations below.
                    # Here, we only set the parameters for them.
                    random_crop = (crop_height, crop_width, max_crop_and_resize[2], max_crop_and_resize[3])
                    resize = (max_crop_and_resize[0], max_crop_and_resize[1])

                if full_crop_and_resize:

                    resize_aspect_ratio = full_crop_and_resize[1] / full_crop_and_resize[0]

                    if img_width < img_height:
                        crop_height = img_height
                        crop_width = int(round(crop_height * resize_aspect_ratio))
                    else:
                        crop_width = img_width
                        crop_height = int(round(crop_width / resize_aspect_ratio))
                    # The actual cropping and resizing will be done by the random crop and resizing operations below.
                    # Here, we only set the parameters for them.
                    if max_crop_and_resize:
                        p = np.random.uniform(0,1)
                        if p >= (1-full_crop_and_resize[4]):
                            random_crop = (crop_height, crop_width, full_crop_and_resize[2], full_crop_and_resize[3])
                            resize = (full_crop_and_resize[0], full_crop_and_resize[1])

            if train: # During training we need the encoded labels instead of the tools that `batch_y` has
                if ssd_box_encoder is None:
                    raise ValueError("`ssd_box_encoder` cannot be `None` in training mode.")
                y_true = ssd_box_encoder.encode_y(batch_y, diagnostics) # Encode the labels into the `y_true` tensor that the cost function needs

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes.
            #          At this point, all images have to have the same size, otherwise you will get an error during training.
            if train:
                yield (np.array(batch_X), y_true)
            else:
                if not batch_y is None:
                    yield (np.array(batch_X), batch_y, this_filenames)
                else:
                    yield (np.array(batch_X), this_filenames)

    def get_filenames_labels(self):
        '''
        Returns:
            The list of filenames and the list of labels.
        '''
        return self.filenames, self.labels

    def get_n_samples(self):
        '''
        Returns:
            The number of image files in the initialized datum.
        '''
        return len(self.filenames)
