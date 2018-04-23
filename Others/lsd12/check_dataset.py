import cv2

from datum.utils.tools import extract_target_from_xml
from Others.lsd12.label_config import sign_idx_dict, idx_sign_dict


def get_true_id_label(label_name):
    """
    :return: label_id, label_name
    """
    return sign_idx_dict[label_name], idx_sign_dict[sign_idx_dict[label_name]]

dataset_dir = "/Volumes/projects/DataSets/LSD12/"
with open(dataset_dir + "total.txt", "r") as reader:
    for line in reader.readlines():
        line = line.strip()
        image_name = line + ".jpg"
        anno_name = line + ".xml"
        image = cv2.imread(dataset_dir + "JPEGImages/" + image_name)
        anno_list = extract_target_from_xml(dataset_dir + "Annotations/" + anno_name)
        for item in anno_list:
            label_id, label_name = get_true_id_label(item[-1])
            item[-1] = label_name
            xmin, ymin, xmax, ymax = item[:4]
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.imshow("src", image)
        cv2.waitKey()


if __name__ == '__main__':
    pass
