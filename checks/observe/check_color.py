# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/2/28

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from skimage import data

import eagle.utils as eu
from eagle.observe.augmentors.flip import Fliplr
from eagle.observe.augmentors.arithmetic import Add
from eagle.observe.augmentors.color import WithChannels, WithColorspace

TIME_PER_STEP = 10000


def main_WithChannels():
    image = data.astronaut()
    print("image shape:", image.shape)
    print("Press any key or wait %d ms to proceed to the next image." % (TIME_PER_STEP,))

    children_all = [
        ("hflip", Fliplr(1)),
        ("add", Add(50))
    ]

    channels_all = [
        None,
        0,
        [],
        [0],
        [0, 1],
        [1, 2],
        [0, 1, 2]
    ]

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.imshow("aug", image[..., ::-1])
    cv2.waitKey(TIME_PER_STEP)

    for children_title, children in children_all:
        for channels in channels_all:
            aug = WithChannels(channels=channels, children=children)
            img_aug = aug.augment_image(image)
            print("dtype", img_aug.dtype, "averages", np.average(img_aug, axis=tuple(range(0, img_aug.ndim-1))))
            #print("dtype", img_aug.dtype, "averages", img_aug.mean(axis=range(1, img_aug.ndim)))

            # title = "children=%s | channels=%s" % (children_title, channels)
            # img_aug = ia.draw_text(img_aug, x=5, y=5, text=title)

            cv2.imshow("aug", img_aug[..., ::-1]) # here with rgb2bgr
            cv2.waitKey(TIME_PER_STEP)


def main_WithColorspace():
    image = data.astronaut()
    print("image shape:", image.shape)

    aug = WithColorspace(
        from_colorspace="RGB",
        to_colorspace="HSV",
        children=WithChannels(0, Add(50))
    )

    aug_no_colorspace = WithChannels(0, Add(50))

    img_show = np.hstack([
        image,
        aug.augment_image(image),
        aug_no_colorspace.augment_image(image)
    ])

    cv2.namedWindow("aug", cv2.WINDOW_NORMAL)
    cv2.imshow("aug", img_show[..., ::-1])
    cv2.waitKey(TIME_PER_STEP)

if __name__ == "__main__":
    # main_WithChannels()
    main_WithColorspace()
