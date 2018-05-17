# encoding: utf-8
"""
@contact: liuguiyang15@mails.ucas.edu.cn
@file: clip_video.py
@time: 2018/5/2 16:57
"""

# 主要是将吉林一号的视频数据进行裁剪，将图像的尺寸降下来，把没有的区域先去掉

import os
import cv2
from sklearn.utils import shuffle

from datum.utils.tools import extract_target_from_xml


video_names = [
    "large_000013363_total", "large_000014631_total",
    "large_minneapolis_1_total", "large_tunisia_total"]
# video_names = ["large_tunisia_total"]
root_dir_path = "/Volumes/projects/DataSets/CSUVideo/"
src_dir_path = root_dir_path + "吉林一号视频逐帧/"
clip_save_dir_path = root_dir_path + "标注结果图/"
clip_spec_infos = {
    # "large_000013363_total": {
    #     'xmin': 750, 'ymin': 0,
    #     'xmax': 3750, 'ymax': 2700
    # },
    "large_000013363_total": {
        'xmin': 0, 'ymin': 0,
        'xmax': 4096, 'ymax': 3072
    },
    # "large_000014631_total": {
    #     'xmin': 0, 'ymin': 500,
    #     'xmax': 3400, 'ymax': 3050
    # },
    "large_000014631_total": {
        'xmin': 0, 'ymin': 0,
        'xmax': 4096, 'ymax': 3072
    },
    "large_minneapolis_1_total": {
        'xmin': 0, 'ymin': 0,
        'xmax': 4096, 'ymax': 2160
    },
    "large_tunisia_total": {
        'xmin': 0, 'ymin': 0,
        'xmax': 4096, 'ymax': 2160
    }
}


def clipping_video(is_show=False, is_save_anno=True, is_save_image=False, is_save_anno_image=False):
    for video_name in video_names:
        xmin, ymin = clip_spec_infos[video_name]["xmin"], clip_spec_infos[video_name]["ymin"]
        xmax, ymax = clip_spec_infos[video_name]["xmax"], clip_spec_infos[video_name]["ymax"]

        video_image_dir_path = src_dir_path + video_name + "/JPEGImages/"
        anno_image_dir_path = src_dir_path + video_name + "/Annotations/"
        if is_show:
            cv2.namedWindow("src", cv2.WINDOW_NORMAL)
        N = len(os.listdir(video_image_dir_path))
        for image_id in range(1, N+1):
            image_path = video_image_dir_path + "%06d.jpg" % image_id
            anno_path = anno_image_dir_path + "%06d.xml" % image_id
            if not os.path.exists(anno_path):
                print(anno_path)
                continue
            anno_lists = extract_target_from_xml(anno_path)
            print(anno_path, len(anno_lists))

            image = cv2.imread(image_path)
            image = image[ymin:ymax, xmin:xmax]
            if is_save_image:
                cv2.imwrite(
                    clip_save_dir_path + video_name + "/JPEGImages/%06d.jpg" % image_id,
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            for anno in anno_lists:
                a_xmin, a_ymin, a_xmax, a_ymax = anno[:4]
                x1 = a_xmin - xmin
                y1 = a_ymin - ymin
                x2 = a_xmax - xmin
                y2 = a_ymax - ymin
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if is_save_anno_image:
                cv2.imwrite(
                    clip_save_dir_path + video_name + "/JPEGImages/%06d.jpg" % image_id,
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            if is_show:
                cv2.imshow("src", image)
                ch = cv2.waitKey(0)
                if ch == ord('q'):
                    return

            # 存储当前的目标在裁剪过后的图像中的位置信息
            if is_save_anno:
                save_new_anno_file = clip_save_dir_path + video_name + "/Annotations/%06d.txt" % image_id
                with open(save_new_anno_file, "w") as writer:
                    writer.write("x1,y1,x2,y2,label\n")
                    for item in anno_lists:
                        writer.write("{},{},{},{},{}\n".format(*item))


# 随机采样：提取20%的图像数据进行模型训练
def shuffle_samples():
    for video_name in video_names:
        image_dir_path = clip_save_dir_path + video_name + "/JPEGImages/"
        anno_dir_path = clip_save_dir_path + video_name + "/Annotations/"

        images_list = os.listdir(image_dir_path)
        N = len(images_list)
        selected_list = shuffle(images_list)[0:int(0.1 * N)]
        for item in images_list:
            if item not in selected_list:
                anno_name = item.split(".")[0] + ".txt"
                os.remove(image_dir_path + item)
                os.remove(anno_dir_path + anno_name)


def crop_image_by_window():
    pass


if __name__ == '__main__':
    clipping_video(is_show=False, is_save_anno=True, is_save_image=False, is_save_anno_image=True)
    # shuffle_samples()
    pass
