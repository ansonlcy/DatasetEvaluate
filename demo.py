import evaluate_utils
import sample_data
import os

os.environ['PATH'] = os.environ['PATH'] + ':/usr/local/cuda/bin'


def main():

    """
    -------------  Demo 1: The statistics complexity ----------------------
    """
    # this is the demo if cal the statistics complexity of the sample data
    # the strcut of the sample data is show in README.md


    # Drive length
    drive_length = evaluate_utils.get_valid_miles(sample_data.sample_suscape_traj, frame_per_scene=40)
    print(f"Drive length: {drive_length:.2f} km")

    # Cover area
    cover_area = evaluate_utils.get_area_cover(sample_data.sample_suscape_traj, lidar_radius=50)
    print(f"Cover area: {cover_area:.2f} km^2")

    # Rotation entropy
    rotation_entropy = evaluate_utils.get_frame_rotation_entropy(sample_data.sample_suscape_ori_data)
    print(f"Rotation entropy: {rotation_entropy:.2f}")

    # valid points
    valid_points_idx = evaluate_utils.get_points_in_box(sample_data.sample_suscape_points)
    print(f"There are {len(valid_points_idx)} points in sample box")

    # Occluded
    occluded = evaluate_utils.get_occlusion_level(sample_data.sample_suscape_points, valid_points_idx)
    print(f"Occluded level is : {occluded:.2f}")

    # Traffic participants density
    density = evaluate_utils.get_commmon_density(sample_data.sample_suscape_ori_data)
    print(f"Traffic participants density: {density:.2f}")

    # Time entropy
    time_entropy = evaluate_utils.get_time_entropy(sample_data.sample_suscape_time)
    print(f"Time entropy: {time_entropy:.2f}")

    # Ego speed entropy
    speed_entropy = evaluate_utils.get_speed_entropy(sample_data.sample_suscape_speed)
    print(f"Ego speed entropy: {speed_entropy:.2f}")

    # Scale entropy
    scale_entropy = evaluate_utils.get_box_scale_s_entropy(sample_data.sample_suscape_scale_ori_data)
    print(f"Scale entropy: {scale_entropy:.2f}")

    # Category entropy
    category_entropy = evaluate_utils.get_category_entropy(sample_data.sample_suscape_ori_data)
    print(f"Category entropy: {category_entropy:.2f}")


    """
    -------------  Demo 2: The frame similarity evaluation ----------------------
    """
    # this is the demo if cal the frame similarity of the sample data
    # the strcut of the sample data is show in README.md
    
    # Frame similarity
    smilarity_metrix = evaluate_utils.get_similarity_among_frames(sample_data.sample_kitti_det)
    print(f"Frame similarity metrix: \n{smilarity_metrix}")



if __name__ == "__main__":
    main()
