# Copyright (c) 2009 IW.
# All rights reserved.
#
# Author: liuguiyang <liuguiyangnwpu@gmail.com>
# Date:   2018/1/3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2

video_filepath = "/Volumes/projects/VIRAT_VIDEO_DATASETS/VERSION02/train/09152008flight2tape2_1.avi"
cap = cv2.VideoCapture(video_filepath)
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


if __name__ == "__main__":
	pass