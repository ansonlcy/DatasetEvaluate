from math import sin, cos

import numpy as np


def entropy(frequency):
    fre = np.array(frequency)
    fre = fre[fre != 0]
    en = np.sum(-fre * np.log2(fre))
    return en


def euler_angle_to_rotate_matrix_3by3(eu):
    """
    get the rotation matrix according to the "rotation" in the calibration file
    :param eu: euler angle in the calibration file
    :param order: rotation order
    :return:
    """

    # Calculate rotation about x axis
    R_x = np.array([
        [1, 0, 0],
        [0, cos(eu[0]), -sin(eu[0])],
        [0, sin(eu[0]), cos(eu[0])]
    ])
    # Calculate rotation about y axis
    R_y = np.array([
        [cos(eu[1]), 0, sin(eu[1])],
        [0, 1, 0],
        [-sin(eu[1]), 0, cos(eu[1])]
    ])
    # Calculate rotation about z axis
    R_z = np.array([
        [cos(eu[2]), -sin(eu[2]), 0],
        [sin(eu[2]), cos(eu[2]), 0],
        [0, 0, 1]])
    R = np.matmul(R_x, np.matmul(R_y, R_z))
    # R = np.matmul(R_z, np.matmul(R_y, R_x))
    return R
