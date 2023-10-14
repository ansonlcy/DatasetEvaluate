import numpy as np
import matplotlib.pyplot as plt
from evaluate_utils.common_utils import entropy
from collections import defaultdict


def draw_data_density(fig_name, x_label, y_label, data_list, bins):
    plt.figure(dpi=300)
    plt.title(fig_name)
    sns.histplot(data=data_list, bins=bins, stat="probability")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    # stas = plt.hist(x=data_list, bins=bins, density=1)
    return


def cal_scale_entropy(scale_data, bins):
    plt.cla()
    stats = plt.hist(x=scale_data, bins=bins, density=1)
    frequency = stats[0] * np.diff(stats[1])
    en = entropy(frequency)
    return en



def create_scale_data(scale_ori_data, save_path=None):
    box_s_dict = defaultdict(list)
    box_l_dict = defaultdict(list)

    for frame_key in scale_ori_data.keys():
        frame = scale_ori_data[frame_key]
        for obj_type in frame.keys():
            box_list = frame[obj_type]
            for box in box_list:
                box_scale = box["box_wlh"]
                s = np.abs(box_scale[0] * box_scale[1])  # some wrong in suscape scale is <0
                # if obj_type == ("Car" or "Truck"):
                #     if s < 2:
                #         print(obj_type, frame_key, s)
                l = np.abs(box_scale[1])
                box_s_dict[obj_type].append(s)
                box_l_dict[obj_type].append(l)
    scale_dict = {"s": box_s_dict, "l": box_l_dict}

    if save_path is not None:
        np.save(save_path, scale_dict)

    return scale_dict


def get_box_scale_s_entropy(scale_ori_data):
    scale_dict = create_scale_data(scale_ori_data)
    scale_s_dict = scale_dict["s"]
    all_list = []
    for value in scale_s_dict.values():
        all_list += value
    all_list = np.array(all_list)
    all_list = all_list[all_list < 60]
    scale_s_entropy = cal_scale_entropy(all_list, 100)
    return scale_s_entropy


if __name__ == "__main__":
    # for dataset_name in ["suscape", "nuscenes", "lyft", "kitti"]:
    for dataset_name in ["suscape"]:
        scale_ori_data = np.load(f"../data_source/stat_complexity/sample_{dataset_name}_scale_ori_data.npy", allow_pickle=True).item()
        print(get_box_scale_s_entropy(scale_ori_data))

