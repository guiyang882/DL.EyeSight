import cv2

from datum.utils.tools import extract_target_from_xml
from Others.lsd12.label_config import sign_idx_dict, idx_sign_dict


dataset_dir = "/Volumes/projects/DataSets/LSD12/"


def get_true_id_label(label_name):
    """
    :return: label_id, label_name
    """
    return sign_idx_dict[label_name], idx_sign_dict[sign_idx_dict[label_name]]


def disp_image():
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


def convert_standard():
    output_path = dataset_dir + "train_data_list.txt"
    write_handler = open(output_path, "w")

    with open(dataset_dir + "train.txt", "r") as reader:
        for line in reader.readlines():
            line = line.strip()
            image_path = dataset_dir + "JPEGImages/" + line + ".jpg"
            anno_path = dataset_dir + "Annotations/" + line + ".xml"
            anno_list = extract_target_from_xml(anno_path)
            anno_str_list = []
            for item in anno_list:
                label_id, label_name = get_true_id_label(item[-1])
                item[-1] = label_id - 1
                item = [str(cell) for cell in item]
                anno_str_list.append(" ".join(item))
            anno_info = " ".join(anno_str_list)
            write_handler.write(image_path + " " + anno_info + "\n")
    write_handler.close()
    print("save the convert_data info to ", output_path)


if __name__ == '__main__':
    convert_standard()
