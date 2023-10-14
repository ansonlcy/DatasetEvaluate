import numpy as np

sample_suscape_ori_data = np.load("./data_source/stat_complexity/sample_suscape_ori_data.npy", allow_pickle=True).item()
sample_suscape_scale_ori_data = np.load("./data_source/stat_complexity/sample_suscape_scale_ori_data.npy",
                                        allow_pickle=True).item()
sample_suscape_points = np.load("./data_source/stat_complexity/sample_suscape_points.npy", allow_pickle=True).item()
sample_suscape_speed = np.load("./data_source/stat_complexity/sample_suscape_speed.npy", allow_pickle=True)
sample_suscape_time = np.load("./data_source/stat_complexity/sample_suscape_time.npy", allow_pickle=True)
sample_suscape_traj = np.load("./data_source/stat_complexity/sample_suscape_traj.npy", allow_pickle=True)

sample_kitti_det = np.load("./data_source/scene_similarity/sample_kitti_det.npy", allow_pickle=True).item()
None