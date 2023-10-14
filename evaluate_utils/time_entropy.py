import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from evaluate_utils.common_utils import entropy
import pandas as pd


def time_heatmap(time_data):
    t_map = np.zeros((24, 60))
    for hour in time_data:
        h = int(np.floor(hour))
        m = int((hour-h)*60)
        t_map[h, m] += 1

    sns.heatmap(data=t_map)
    plt.show()
    return


def get_time_entropy(time_list, interval=0.01):
    r_slice = np.arange(0, 24, interval)
    c = pd.cut(time_list, r_slice)
    sum_num = np.sum(c.value_counts().values)
    frequency = [i / sum_num for i in c.value_counts().values]
    en = entropy(frequency)
    return en


if __name__ == "__main__":
    for dataset_name in ["suscape"]:
        print(dataset_name)
        time_data = np.load(f"../data_source/stat_complexity/sample_{dataset_name}_time.npy", allow_pickle=True)
        # y = np.zeros_like(time_data)
        # plt.figure(dpi=300)
        # plt.scatter(x=time_data, y=y, s=0.5)
        # plt.show()
        # time_heatmap(time_data)
        # print(np.std(time_data))
        print(get_time_entropy(time_data, 0.01))


