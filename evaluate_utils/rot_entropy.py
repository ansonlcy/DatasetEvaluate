import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from evaluate_utils.common_utils import entropy


def normalize_r(theta):
    theta = theta * 180 / np.pi
    theta = theta - np.floor(theta / 360.0) * 360
    return theta


def cal_rotation_entropy(r_list, interval=15):
    r_slice = range(0, 361, interval)
    c = pd.cut(r_list, r_slice)
    sum_num = np.sum(c.value_counts().values)
    frequency = [i / sum_num for i in c.value_counts().values]
    en = entropy(frequency)
    return en


def create_rotation_data(ori_dict):
    frame_rotation = defaultdict(list)
    type_rotation = defaultdict(list)

    for frame_key in ori_dict.keys():
        frame = ori_dict[frame_key]
        for obj_type_key in frame.keys():
            box_list = frame[obj_type_key]
            for box in box_list:
                frame_rotation[frame_key].append(box["rotation_z"])
                type_rotation[box["obj_type"]].append(box["rotation_z"])

    return frame_rotation, type_rotation


def get_rotation_entropy_dict(ori_dict, interval, save_path=None):
    frame_entropy_list = []
    type_entropy_dict = {}

    frame_rotation, type_rotation = create_rotation_data(ori_dict)
    for values in frame_rotation.values():
        r_list = np.clip(normalize_r(np.array(values)), 0.01, 360)
        frame_entropy = cal_rotation_entropy(r_list, interval=interval)
        frame_entropy_list.append(frame_entropy)

    for obj_type_key in type_rotation.keys():
        r_list = normalize_r(np.array(type_rotation[obj_type_key]))
        type_entropy = cal_rotation_entropy(r_list, interval=interval)
        type_entropy_dict[obj_type_key] = type_entropy

    rotation_entropy_dict = {
        "frame_entropy_list": frame_entropy_list,
        "type_entropy_dict": type_entropy_dict
    }

    if save_path is not None:
        np.save(save_path, rotation_entropy_dict)

    return rotation_entropy_dict


def get_frame_rotation_entropy(ori_dict, interval=15):
    rotation_entropy_dict = get_rotation_entropy_dict(ori_dict, interval=interval)
    frame_mean_entropy = np.mean(rotation_entropy_dict["frame_entropy_list"])
    return frame_mean_entropy



if __name__ == "__main__":

    # for dataset_name in ["suscape", "nuscenes", "lyft", "kitti"]:
    for dataset_name in ["suscape"]:
        print(dataset_name)
        ori_data = np.load(f"../data_source/stat_complexity/sample_{dataset_name}_ori_data.npy", allow_pickle=True).item()
        # get_rotation_entropy_dict(ori_data, interval=15, save_path=f"./{dataset_name}_entropy.npy")
        print(get_frame_rotation_entropy(ori_data, interval=15))
