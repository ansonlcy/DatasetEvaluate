import math
import numpy as np
from collections import defaultdict
from evaluate_utils.common_utils import euler_angle_to_rotate_matrix_3by3


def get_points_grid_index(points, point_grid_size=1):
    points_reverse = np.transpose(points)
    points_grids = np.floor(points_reverse / point_grid_size)

    tree = lambda: defaultdict(tree)
    points_index = tree()

    for idx, p in enumerate(points_grids):

        if len(points_index[p[0]][p[1]][p[2]]) == 0:
            points_index[p[0]][p[1]][p[2]] = []

        points_index[p[0]][p[1]][p[2]].append(idx)

    return points_index


def get_rotate_matrix(box_center, box_rotation):
    """
    get the rotation matrix from the label info rotation

    :return:  shape 4*4
    """
    R = euler_angle_to_rotate_matrix_3by3(box_rotation)
    trans = box_center.reshape([-1, 1])
    R = np.concatenate([R, trans], axis=-1)
    R = np.concatenate([R, np.array([0, 0, 0, 1]).reshape([1, -1])], axis=0)
    return R


def get_box_corners(box_center, box_rotation, box_scale):
    x = box_scale[0]
    y = box_scale[1]
    z = box_scale[2]
    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    # front-left-bottom front-right-bottom front-right-top front-left-top rear-left-bottom rear-right-bottom rear-right-top rear-left-top
    x_corners = x / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = y / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = z / 2 * np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    p_corners = np.vstack((x_corners, y_corners, z_corners))
    p_corners = np.concatenate([p_corners.T, np.ones(8).reshape(8, -1)], axis=1)
    # translate the corers to the Lidar coordinate frame
    p_corners = np.matmul(get_rotate_matrix(box_center, box_rotation), np.transpose(p_corners))
    return p_corners


def get_candidate_points_around_box(box_center, box_rotation, box_scacle, points_grid_index, grid_size=1,
                                    box_corners=None):
    if box_corners is None:
        corners = get_box_corners(box_center, box_rotation, box_scacle)
    else:
        corners = box_corners
    b_max = np.max(corners, axis=1)
    b_min = np.min(corners, axis=1)
    indices = []
    for x in np.arange(np.floor(b_min[0] / grid_size), np.floor(b_max[0] / grid_size) + 1, 1):
        for y in np.arange(np.floor(b_min[1] / grid_size), np.floor(b_max[1] / grid_size) + 1, 1):
            for z in np.arange(np.floor(b_min[2] / grid_size), np.floor(b_max[2] / grid_size) + 1, 1):
                tmp = points_grid_index[x][y][z]
                if type(tmp) is list:
                    indices += (points_grid_index[int(x)][int(y)][int(z)])

    return indices


def get_points_in_box_2(points, box_center, box_rotation, box_scale, rot_mat=None):
    if rot_mat is None:
        rot_mat = euler_angle_to_rotate_matrix_3by3(box_rotation)
    ground_level = 0.3
    if box_scale[2] < 2:
        ground_level = box_scale[2] * 0.15

    p_all = np.matmul(np.transpose(rot_mat), points - box_center.reshape(3, -1))

    p_filter = (p_all[0, :] > - box_scale[0] / 2) & (p_all[0, :] < box_scale[0] / 2) & (
            p_all[1, :] > - box_scale[1] / 2) & (p_all[1, :] < box_scale[1] / 2)
    p_filter = p_filter & (p_all[2, :] > - box_scale[2] / 2 + ground_level) & (p_all[2, :] < box_scale[2] / 2)

    indices = np.argwhere(p_filter > 0)

    return indices


def object_plane_visible_ratio(box_scale, points, axies, ground_level):
    """
    axies: [0,2], or [1, 2]
    """

    scale = np.array([box_scale[0], box_scale[1], box_scale[2]])
    grid_num = np.array([10, 10, 6])
    points = points[:, axies]
    scale = scale[axies]
    grid_num = grid_num[axies]

    grid = scale / grid_num
    # print(box)
    # print('grid', grid)

    pts = (points / grid) + grid_num / 2
    # pts = pts[0:3, :]
    # print(pts[0:3,:])
    ind = pts.astype(int)
    # print(ind)

    ind[ind >= grid_num] -= 1

    # print(ind)
    occupation = np.zeros(grid_num)

    occupation[ind[:, 0], ind[:, 1]] = 1
    # print(box['obj_id'], axies, occupation)

    # only check center part, since some object has extremities.

    center_occupation = occupation[2:8, 1:5]

    return np.sum(center_occupation) / (center_occupation.shape[0] * center_occupation.shape[1])


def object_occluded(points_in_box, ground_level, box_center, box_rotation, box_scale):
    """
    Note that the "points_in_box" is the points (in the box) that translation to the origin of lidar coordinate

    if for any plane the points occupy 2/3 parts of it,
    it's considered not occluded.
    """
    dim_occlusion_ratio = 0.7
    area_occlusion_ratio = 0.4

    min_pts = np.min(points_in_box, axis=0)
    max_pts = np.max(points_in_box, axis=0)

    # 0.2 for annotation error: box may be larger than object.
    x_visible_ratio = (max_pts[0] - min_pts[0] + 0.2) / box_scale[0]
    y_visible_ratio = (max_pts[1] - min_pts[1] + 0.2) / box_scale[1]
    z_visible_ratio = (max_pts[2] - min_pts[2] + 0.2 + ground_level) / box_scale[2]

    xz_area_visible_ratio = object_plane_visible_ratio(box_scale, points_in_box, [0, 2], ground_level)
    yz_area_visible_ratio = object_plane_visible_ratio(box_scale, points_in_box, [1, 2], ground_level)

    view_angle = box_rotation[2] - math.atan2(box_center[1], box_center[0])
    view_angle %= math.pi * 2  # [0, 2pi)
    if view_angle > math.pi:
        view_angle -= math.pi

    # [0 , pi]
    if view_angle > math.pi / 2:
        view_angle = math.pi - view_angle

    area = 1.0
    # # [0, pi/2]
    # if view_angle > 15 / 180 * math.pi:
    #     # x is ok
    #     if x_visible_ratio < dim_occlusion_ratio:
    #         return True
    #     if xz_area_visible_ratio < area_occlusion_ratio:
    #         return True
    #
    # if view_angle < 75 / 180 * math.pi:
    #     # y is ok
    #     if y_visible_ratio < dim_occlusion_ratio:
    #         return True
    #     if yz_area_visible_ratio < area_occlusion_ratio:
    #         return True
    #
    # if z_visible_ratio < dim_occlusion_ratio:
    #     return True
    if view_angle <= 15 / 180 * math.pi:
        return yz_area_visible_ratio

    elif 15 / 180 * math.pi < view_angle < 75 / 180 * math.pi:
        return max(xz_area_visible_ratio, yz_area_visible_ratio)

    else:
        return xz_area_visible_ratio


def _get_points_in_box(points, box_center, box_rotation, box_scacle, rot_mat=None, corners=None):
    """
    get the points in the box
    :param points: ndarray, shape (3, N)
    :param box_center: ndarray, shape (3,),  (x,y,z) in meter
    :param box_rotation: ndarray, shape (3,), in radian
    :param box_scacle: ndarray shape (3,), box scale(meter) in x,y,z axis
    :param rot_mat: ndarray, shape (3,3), use the specific rotation matrix
    :param corners: ndarray, shape (3,8), if we have the box corners, we can use it to speed up the process
    :return: ndarray, shape (M, 1), the index of points in box
    """
    if rot_mat is None:
        rot_mat = euler_angle_to_rotate_matrix_3by3(box_rotation)

    points_grid_index = get_points_grid_index(points)

    # the translate is (0,0,0)
    candidate_index = get_candidate_points_around_box(box_center, box_rotation, box_scacle, points_grid_index,
                                                      box_corners=corners)
    candidate_points = points[:, candidate_index]

    p_all = np.matmul(np.transpose(rot_mat), candidate_points - box_center.reshape(3, -1))

    indices = []

    r = (np.abs(p_all) - (box_scacle.reshape(3, -1) / 2 + 0.01)).T

    for idx, p in enumerate(r):
        if (p[0] > 0) or (p[1] > 0) or (p[2] > 0):
            continue
        indices.append(candidate_index[idx])

    indices = np.expand_dims(indices, axis=1)

    return indices


def get_points_in_box(det_dict, rot_mat=None, corners=None):
    points = det_dict["points"]
    box_center = det_dict["box"]["center"]
    box_rotation = det_dict["box"]["rotation"]
    box_scale = det_dict["box"]["scale"]
    return _get_points_in_box(points, box_center, box_rotation, box_scale, rot_mat, corners)


def _get_occlusion_level(points, points_in_box_index, box_center, box_rotation, box_scale, rot_mat=None):
    """
    charge the occlusion level of the box
    :param points: ndarray, shape (3, N), points in lidar coordinate
    :param points_in_box_index: ndarray, shape (M, 1), the index of points in box
    :param box_center:
    :param box_rotation:
    :param box_scale:
    :param rot_mat:
    :return: The occlusion level of the box, e.g. if level is 0.7, means 70% of the box is occluded
    """
    points_in_box = np.squeeze(points[:, points_in_box_index], 2)
    if rot_mat is None:
        rot_mat = euler_angle_to_rotate_matrix_3by3(box_rotation)

    ground_level = 0.3
    if box_scale[2] < 2:
        ground_level = box_scale[2] * 0.15

    p_all = np.matmul(np.transpose(rot_mat), points_in_box - box_center.reshape(3, -1))

    level = object_occluded(p_all.T, ground_level, box_center, box_rotation, box_scale)
    return level


def get_occlusion_level(det_dict, indices):
    points = det_dict["points"]
    box_center = det_dict["box"]["center"]
    box_rotation = det_dict["box"]["rotation"]
    box_scale = det_dict["box"]["scale"]
    return _get_occlusion_level(points, indices, box_center, box_rotation, box_scale)


if __name__ == "__main__":

    points_dict = np.load("../data_source/stat_complexity/sample_suscape_points.npy", allow_pickle=True).item()


    indices = get_points_in_box(points_dict)
    level = get_occlusion_level(points_dict, indices)
    print(f"{len(indices)} points in box")
    print(f"box occluded level is {level:.2f}")
