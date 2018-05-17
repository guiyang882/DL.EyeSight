# encoding: utf-8
"""
@contact: liuguiyang15@mails.ucas.edu.cn
@file: prepare_trainsamples.py
@time: 2018/5/17 13:05
"""

# 将吉林一号卫星数据按照指定的格式进行组织
import os
import random
import cv2


train_video = ["large_000014631_total", "large_minneapolis_1_total", "large_tunisia_total"]
anno_dir_prefix = "/Volumes/projects/DataSets/CSUVideo/video_with_annotation/"
image_dir_prefix = "/Volumes/projects/DataSets/CSUVideo/src_video_frame/"
save_dir_prefix = "/Volumes/projects/DataSets/CSUVideo/300x300/"

SUB_IMG_WID, SUB_IMG_HEI, SUB_OVERLAP = 300, 300, 80


def twoboxes_overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def crop_image(image_path, anno_path, video_name, image_name):
    target_annos = []
    with open(anno_path, "r") as reader:
        cnt = 0
        for line in reader:
            cnt += 1
            if cnt == 1:
                continue
            line = list(map(int, line.strip().split(",")[:-1]))
            target_annos.append(line)

    def select_subimage_anno(w, h):
        select_box = []
        for box in target_annos:
            x1, y1, x2, y2 = box
            x11, y11 = x1 - w, y1 - h
            x22, y22 = x2 - w, y2 - h
            gx1, gy1 = w, h
            gx2, gy2 = w + SUB_IMG_WID, h + SUB_IMG_HEI
            overlap_area = twoboxes_overlap(box, [gx1, gy1, gx2, gy2])
            if overlap_area <= 0:
                continue
            new_box = [max(0, x11), max(0, y11), min(x22, SUB_IMG_WID), min(y22, SUB_IMG_HEI)]
            if overlap_area / ((x22 - x11) * (y22 - y11)) >= 0.7:
                select_box.append(new_box)
        return select_box

    image_data = cv2.imread(image_path)
    H, W = image_data.shape[:2]
    cnt = 0
    for h in range(0, H, SUB_IMG_HEI-SUB_OVERLAP):
        for w in range(0, W, SUB_IMG_WID-SUB_OVERLAP):
            if h + SUB_IMG_HEI >= H:
                h = H - SUB_IMG_HEI
            if w + SUB_IMG_WID >= W:
                w = W - SUB_IMG_WID
            cnt += 1
            sub_image = image_data[h:h+SUB_IMG_HEI, w:w+SUB_IMG_WID]
            select_annos = select_subimage_anno(w, h)
            if len(select_annos) == 0:
                continue
            # print(len(select_annos), select_annos)
            # for box in select_annos:
            #     x1, y1, x2, y2 = box
            #     cv2.rectangle(sub_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # cv2.imshow("src", sub_image)
            # cv2.waitKey()
            image_name = image_name.split(".")[0]
            if not os.path.isdir(save_dir_prefix + video_name + "/JPEGImages/"):
                os.makedirs(save_dir_prefix + video_name + "/JPEGImages/")
            if not os.path.isdir(save_dir_prefix + video_name + "/Annotations/"):
                os.makedirs(save_dir_prefix + video_name + "/Annotations/")
            save_image_path = save_dir_prefix + video_name + "/JPEGImages/{}_{}_{}.jpg".format(image_name, w, h)
            save_anno_path = save_dir_prefix + video_name + "/Annotations/{}_{}_{}.txt".format(image_name, w, h)
            cv2.imwrite(save_image_path, sub_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            with open(save_anno_path, "w") as writer:
                for box in select_annos:
                    writer.write(",".join(map(str, box)) + "\n")

# for video_name in train_video:
#     image_dir_path = image_dir_prefix + video_name + "/JPEGImages/"
#     anno_dir_path = anno_dir_prefix + video_name + "/Annotations/"
#     anno_list = os.listdir(anno_dir_path)
#     random.shuffle(anno_list)
#     anno_list = random.sample(anno_list, int(len(anno_list) * 0.06))
#     for anno_name in anno_list:
#         anno_path = anno_dir_path + anno_name
#         image_path = image_dir_path + anno_name.replace("txt", "jpg")
#         print(anno_path)
#         print(image_path)
#         crop_image(image_path, anno_path, video_name, anno_name)


train_sample_path = save_dir_prefix + "train_samples.txt"
writer = open(train_sample_path, "w")
for video_name in train_video:
    image_dir_path = save_dir_prefix + video_name + "/JPEGImages/"
    anno_dir_path = save_dir_prefix + video_name + "/Annotations/"
    anno_list = os.listdir(anno_dir_path)
    for anno_name in anno_list:
        anno_path = anno_dir_path + anno_name
        image_path = image_dir_path + anno_name.replace("txt", "jpg")
        anno_detail = ""
        with open(anno_path, "r") as reader:
            anno_info = []
            for line in reader:
                line = line.strip().split(",") + ["0"]
                anno_info.append(" ".join(line))
            anno_detail = " ".join(anno_info)
        writer.write("{} {}\n".format(image_path, anno_detail))
