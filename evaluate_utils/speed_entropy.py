import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from evaluate_utils.common_utils import entropy


def draw_data_density(fig_name, x_label, y_label, data_list, bins):

    plt.figure(dpi=300)
    plt.title(fig_name)
    sns.histplot(data=data_list, bins=bins, stat="probability")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    # stas = plt.hist(x=data_list, bins=bins, density=1)
    return


def get_speed_entropy(speed_data, bins=50):
    plt.cla()
    stats = plt.hist(x=speed_data, bins=bins, density=1)
    frequency = stats[0] * np.diff(stats[1])
    en = entropy(frequency)

    return en



if __name__ == "__main__":

    all_speed = {}
    for dataset_name in ["suscape"]:
        print(dataset_name)
        speed_data = np.load(f"../data_source/stat_complexity/sample_{dataset_name}_speed.npy", allow_pickle=True)
        speed_data = speed_data * 3.6
        all_speed[dataset_name] = speed_data
        print(f"Speed mean: {np.mean(speed_data)}, std: {np.std(speed_data)}")
        draw_data_density(fig_name=f"{dataset_name}_speed", x_label="km/h", y_label="percent", data_list=speed_data,
                           bins=50)
        en = get_speed_entropy(speed_data, 50)
        print(f"Speed entropy: {en}")
