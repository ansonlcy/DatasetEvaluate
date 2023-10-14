import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def get_area_cover(traj, lidar_radius):
    """

    :param traj: ndarray(N, 2)
    :param lidar_radius: int lidar cover radius
    :return: area km^2
    """
    traj_min = np.min(traj, axis=0)
    size = np.max(traj, axis=0) - traj_min
    pad_num = lidar_radius + 50
    x_num = int(size[0]+pad_num)
    y_num = int(size[1]+pad_num)
    area = np.zeros((x_num, y_num))

    for p in traj:
        grid_index = ((p - traj_min) + 200).astype(np.int)
        area[(grid_index[0]-lidar_radius):(grid_index[0]+lidar_radius), (grid_index[1]-lidar_radius):(grid_index[1]+lidar_radius)] = 1

    area_total = np.sum(area)/1000000

    # area = cv2.resize(area, (5000, 5000))
    # # cv2.imshow("t",area)
    # # cv2.waitKey(0)
    # cv2.imwrite("./t.png", area)

    return area_total


def get_valid_miles(traj, frame_per_scene):
    """

    :param traj: ndarray(N, 2) *note that N = scene_num * frame_per_scene
    :param frame_per_scene: The number of frames in each scene
    :return:
    """

    interval = traj[:-1] - traj[1:]
    zero_slice = np.arange(frame_per_scene-1, len(traj)-frame_per_scene, frame_per_scene)
    interval[zero_slice] = [0, 0]
    mileage = np.sum(np.linalg.norm(interval, axis=1)) / 1000

    return mileage


def plot_traj(traj):
    plt.figure(dpi=300)
    plt.scatter(traj[:, 0], traj[:, 1], s=0.1)
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()
    return


if __name__ == "__main__":
    traj_suscape = np.load("../data_source/stat_complexity/sample_suscape_traj.npy", allow_pickle=True)
    print("***suscape****")
    print(f'Total miles: {get_valid_miles(traj_suscape, 40)} km')
    print(f"Total cover: {get_area_cover(traj_suscape, 100)} km^2")
    plot_traj(traj_suscape)

    # None









