import numpy as np
from collections import defaultdict


def get_common_density_dict(ori_data):
    type_num_dict = defaultdict(int)

    for frame in ori_data.values():
        for obj_type_key in frame.keys():
            box_list = frame[obj_type_key]
            type_num_dict[obj_type_key] += len(box_list)

    frame_num = len(ori_data)
    for key in type_num_dict.keys():
        type_num_dict[key] /= frame_num

    commom_dict = {key: type_num_dict[key] for key in type_num_dict.keys() if type_num_dict[key] > 1}

    return commom_dict


def get_commmon_density(ori_data):
    common_dict = get_common_density_dict(ori_data)
    common_density = np.sum([value for value in common_dict.values()])
    return common_density


if __name__ == "__main__":
    for dataset_name in ["suscape"]:
        print(dataset_name)
        ori_data = np.load(f"../data_source/stat_complexity/sample_{dataset_name}_ori_data.npy", allow_pickle=True).item()
        common_density = get_commmon_density(ori_data)
        print(common_density)
