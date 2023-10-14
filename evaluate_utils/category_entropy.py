import numpy as np
from collections import defaultdict
from evaluate_utils.common_utils import entropy
from collections import Counter


def get_category_entropy(ori_data):

    category_dict = defaultdict(list)
    for frame in ori_data.values():
        for obj_type in frame.keys():
            category_dict[obj_type].append(len(frame[obj_type]))

    cate_list = []

    for value in category_dict.values():
        cate_list.append(np.sum(value))

    cate_list = np.array(cate_list)
    cate_list_fre = cate_list / np.sum(cate_list)

    en = entropy(cate_list_fre)


    return en


def create_cate_num_figure_data(save_path):
    save_dict = defaultdict()
    for dataset_name in ["suscape", "nuscenes", "lyft", "kitti"]:
        print(dataset_name)
        ori_data = np.load(f"../data/{dataset_name}_ori_data.npy", allow_pickle=True).item()
        num_list = []
        for frame in ori_data.values():
            num_list.append(len(frame))
        # cout = Counter(num_list)
        save_dict[dataset_name] = num_list

    np.save(save_path, save_dict)


def create_box_num_figure_data(save_path):
    save_dict = defaultdict()
    for dataset_name in ["suscape", "nuscenes", "lyft", "kitti"]:
        print(dataset_name)
        ori_data = np.load(f"../data/{dataset_name}_ori_data.npy", allow_pickle=True).item()
        num_list = []
        for frame in ori_data.values():
            frame_sum = 0
            for obj_type in frame.keys():
                frame_sum += len(frame[obj_type])
            num_list.append(frame_sum)
        # cout = Counter(num_list)
        save_dict[dataset_name] = num_list

    np.save(save_path, save_dict)



if __name__ == "__main__":
    for dataset_name in ["suscape"]:
        print(dataset_name)
        ori_data = np.load(f"../data_source/stat_complexity/sample_{dataset_name}_ori_data.npy", allow_pickle=True).item()
        en = get_category_entropy(ori_data)
        print(en)



