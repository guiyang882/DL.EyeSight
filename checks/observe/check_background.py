# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/3/1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage import data
import matplotlib.pyplot as plt

import eagle.utils as eu
from eagle.observe.base.meta import Sequential
from eagle.observe.augmentors.flip import Fliplr, Flipud
from eagle.observe.base.basetype import KeyPoint, KeyPointsOnImage
from eagle.observe.base.basebatch import Batch, BatchLoader, BackgroundAugmentor


def main():
    augseq = Sequential([
        Fliplr(0.5),
        Flipud(0.5)
    ])

    print("------------------")
    print("augseq.augment_batches(batches, background=True)")
    print("------------------")
    batches = list(load_images())
    batches_aug = augseq.augment_batches(batches, background=True)
    images_aug = []
    keypoints_aug = []
    for batch_aug in batches_aug:
        images_aug.append(batch_aug.images_aug)
        keypoints_aug.append(batch_aug.keypoints_aug)
    grid = draw_grid(images_aug, keypoints_aug)
    print(grid.shape)
    plt.imshow(grid)
    plt.show()

    print("------------------")
    print("augseq.augment_batches(batches, background=True) -> only images")
    print("------------------")
    batches = list(load_images())
    batches = [batch.images for batch in batches]
    batches_aug = augseq.augment_batches(batches, background=True)
    images_aug = []
    keypoints_aug = None
    for batch_aug in batches_aug:
        images_aug.append(batch_aug)
    plt.imshow(draw_grid(images_aug, keypoints_aug))
    plt.show()

    print("------------------")
    print("BackgroundAugmenter")
    print("------------------")
    batch_loader = BatchLoader(load_images)
    bg_augmenter = BackgroundAugmentor(batch_loader, augseq)
    images_aug = []
    keypoints_aug = []
    while True:
        print("Next batch...")
        batch = bg_augmenter.get_batch()
        if batch is None:
            print("Finished.")
            break
        images_aug.append(batch.images_aug)
        keypoints_aug.append(batch.keypoints_aug)
    plt.imshow(draw_grid(images_aug, keypoints_aug))
    plt.show()


def load_images():
    batch_size = 4
    astronaut = data.astronaut()
    astronaut = eu.imresize_single_image(astronaut, (64, 64))
    kps = KeyPointsOnImage([KeyPoint(x=15, y=25)], shape=astronaut.shape)
    counter = 0
    for i in range(10):
        batch_images = []
        batch_kps = []
        for b in range(batch_size):
            batch_images.append(astronaut)
            batch_kps.append(kps)
            counter += 1
        batch = Batch(
            images=np.array(batch_images, dtype=np.uint8),
            keypoints=batch_kps
        )
        yield batch


def draw_grid(images_aug, keypoints_aug):
    if keypoints_aug is None:
        keypoints_aug = []
        for bidx in range(len(images_aug)):
            keypoints_aug.append([None for _ in images_aug[bidx]])

    images_kps_batches = []
    for bidx in range(len(images_aug)):
        images_kps_batch = []
        for image, kps in zip(images_aug[bidx], keypoints_aug[bidx]):
            if kps is None:
                image_kps = image
            else:
                image_kps = kps.draw_on_image(image, size=5, color=[255, 0, 0])
            images_kps_batch.append(image_kps)
        images_kps_batches.extend(images_kps_batch)

    grid = eu.draw_grid(images_kps_batches, cols=len(images_aug[0]))
    return grid

if __name__ == "__main__":
    main()
