# The Evaluation Metrics of Autonomous Driving Dataset

## 1. Introduction
### 1.1 The Statastics Complexity of Dataset
The complexity of the dataset is mainly reflected in the following aspects:
```python
import numpy as np
import evaluate_utils
```
* Drive length & Cover area
```python
traj_data = np.array([[193002.214671187, 2503275.415500029], ...]) # shape (N, 2), coordinate in UTM
drive_length = evaluate_utils.get_valid_miles(traj_data, frame_per_scene=40)
cover_area = evaluate_utils.get_area_cover(traj_data, lidar_radius=50)
```

* Speed Entropy
```python
speed_data = np.array([1.0, 3.5, ...]) # shape (N,), unit: m/s
speed_entropy = evaluate_utils.get_speed_entropy(speed_data)
```
* Time Entropy
```python
time_data = np.array([12.5, 20.2, ...]) # shape (N,), time in 24 hours [0, 24)
time_entropy = evaluate_utils.get_time_entropy(time_data)
```
* Traffic Participants density
```python
ori_data_dict = {
    '1630376940.000': {
        'Car':[
            {
                "id": "1", # the id of the object
                "obj_type": "Car", # the type of the object
                "distance": 5.0, # the distance between the object and the ego vehicle in meter
                "points": -1, # the number of points in the box, use -1 to represent unknown (will be cal in after process)
                "visibility": -1, # visibility level, use -1 to represent unknown (will be cal in after process)
                "rotation_z": -1.54, # rotion around z-axis in radian
                "box_wlh": [1.5, 1.2, 1.6] # ndarray, shape (3,), (w,l,h) in meter
            },
            ...
        ],
        'Pedestrian':[
            ...
        ],
        'Cyclist':[
            ...
        ],
        ...
    }
    ...
}

density = evaluate_utils.get_commmon_density(ori_data_dict)
```
* Rotation Entropy
```python
# same as ori_data_dict
rotation_entropy = evaluate_utils.get_frame_rotation_entropy(ori_data_dict)
```
* Category Entropy
```python
# same as ori_data_dict
category_entropy = evaluate_utils.get_category_entropy(ori_data_dict)
```
* Scale Entropy
```python
# same as ori_data_dict
scale_entropy = evaluate_utils.get_box_scale_s_entropy(ori_data_dict)
```
* Valid Points & Occluded level
```python
det_dict = {
    "points": points # ndarray shape (3, N)
    "box":{
        "center": center, # ndarray, shape (3,) , (x,y,z) in meter
        "rotation": rotation, # ndarray, shape (3,), in radian
        "scale": scale # ndarray, shape (3,), (l,w,h) in meter
    }
}

valid_points_idx = evaluate_utils.get_points_in_box(det_dict) # points number is len(valid_points_idx)
occluded_level = evaluate_utils.get_occluded_level(det_dict, valid_points_idx)
```

### 1.2 The Similarity of Dataset
* Similarity between frames
```python
simi_det_dict = {
    '000488': [{"label":1, "pos":[18.233434,121.32903]}, ...],
    '000489': [...],
    ...
}
# get the similarity metrix between frames ("000488", "000489",...)
smilarity_metrix = evaluate_utils.get_similarity_among_frames(simi_det_dict)
```

# 2. Installation and Run
Calculate the similarity need to use GraphDot which requires a CUDA Toolkit installation for carrying out GPU computations. To install it, following the instructions on https://developer.nvidia.com/cuda-toolkit.

```python
pip install -r requirements.txt
```
Run ```demo.py``` to get the demo of 11 different evaluation metrics.
