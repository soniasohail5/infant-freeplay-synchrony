import numpy as np
from pathlib import Path
from pkg_resources import parse_version
import os
import tensorflow as tf
import json
import tensorflow_hub as hub

# Folder directories for video frame extraction

# folder_dir = "/mnt/myd/JIM HALPERT BACKUP/3HYPER/3HYPER Video Data/3HYPER FREEPLAY DV AVI"
# dst_dir = "/mnt/myd/JIM HALPERT BACKUP/3HYPER/3HYPER Video Data/3HYPER FREEPLAY DV AVI/extracted frames"

video_duration = 240 # time of video in seconds
fps = 30

def visualize_matplotlib(image, pred, joint_names, joint_edges, save_filepath):
    detections, poses3d, poses2d = pred['boxes'], pred['poses3d'], pred['poses2d']

    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Rectangle
    # plt.switch_backend('TkAgg')

    save_filepath_full = save_filepath + ".png"

    fig = plt.figure(figsize=(10, 5.2))
    image_ax = fig.add_subplot(1, 2, 1)
    image_ax.set_xlim(0, 400)
    image_ax.set_y_lim(0, 400)
    image_ax.imshow(image)
    for x, y, w, h in detections[:, :4]:
        image_ax.add_patch(Rectangle((x, y), w, h, fill=False))

    pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
    pose_ax.view_init(5, -85)
    pose_ax.set_xlim3d(-1500, 1500)
    pose_ax.set_zlim3d(-1500, 1500)
    pose_ax.set_ylim3d(0, 3000)
    pose_ax.set_box_aspect((1, 1, 1))

    # Matplotlib plots the Z axis as vertical, but our poses have Y as the vertical axis.
    # Therefore, we do a 90° rotation around the X axis:
    poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
    for pose3d, pose2d in zip(poses3d, poses2d):
        for i_start, i_end in joint_edges:
            image_ax.plot(*zip(pose2d[i_start], pose2d[i_end]), marker='o', markersize=2)
            pose_ax.plot(*zip(pose3d[i_start], pose3d[i_end]), marker='o', markersize=2)
        image_ax.scatter(*pose2d.T, s=2)
        pose_ax.scatter(*pose3d.T, s=2)

    fig.tight_layout()
    # plt.pause(0.001)
    plt.savefig(save_filepath_full)
    # plt.show()
    plt.close()

def visualize(image, pred, joint_names, joint_edges, save_filepath):
        visualize_matplotlib(image, pred, joint_names, joint_edges, save_filepath)

'''

# Extracting frames from the videos (for testing on MeTRABs model)

for file in sorted(os.listdir(folder_dir))[19:]:
    cap = cv2.VideoCapture(os.path.join(folder_dir, file))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("Opening video file..." + file)

    folder_name = file + " EXTRACTED"

    total_frames = int(fps) * video_duration
    frame_counter = 0
    ret = 1

    while ret or frame_counter < total_frames:

        ret, frame = cap.read()
        try:
            os.makedirs(os.path.join(dst_dir, folder_name), exist_ok=True)
            cv2.imwrite(os.path.join(dst_dir, folder_name, "frame%d.jpg") % frame_counter, frame)

            if ret == 1:
                print("Successfully extracted frame " + str(frame_counter))
                frame_counter += 1

        except cv2.error:
            continue
'''

# Import MeTRABs model

print("Loading model ....")
model = hub.load('https://bit.ly/metrabs_l')
skeleton = 'smpl_24'
joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
frame_counter = 0

folder_path = "/mnt/e/IN-PERSON EXPERIMENT RECORDINGS/3HYPER FREEPLAY/3HYPER DV FREEPLAY/avi/3HYPER FREEPLAY DV AVI/extracted frames"
output_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS"

for folder in sorted(os.listdir(folder_path)):
    print("Opening video folder ..." + folder)
    folder_full_path = os.path.join(folder_path, folder)

    for frame in sorted(os.listdir(folder_full_path)):
        image = tf.image.decode_jpeg(tf.io.read_file(os.path.join(folder_full_path, frame)))
        new_file_name = os.path.splitext(os.path.basename(os.path.join(folder_full_path, frame)))[0] + " keypoints.json"
        img_file_name = os.path.splitext(os.path.basename(os.path.join(folder_full_path, frame)))[0]
        new_file_path = os.path.join(output_dir, folder, new_file_name)
        new_img_path = os.path.join(output_dir, folder, img_file_name)

        
        print("Running inference on " + str(frame))
        pred = model.detect_poses(image, default_fov_degrees=55, skeleton=skeleton)
        pred = tf.nest.map_structure(lambda x: x.numpy(), pred)
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        
        visualize(image, pred, joint_names, joint_edges, new_img_path)

        people = []

        for person in range(pred['poses3d'].shape[0]):
            predict = {'person_id': person, 'poses3d': pred['poses3d'][person, :, :].tolist(), 'poses2d': pred['poses2d'][person, :, :].tolist(), 'boxes': pred['boxes'][person, :].tolist()}
            people.append(predict)

        full_predict  = {"people": people}

        with open(new_file_path, 'w') as new_file:
            json.dump(full_predict, new_file)
            new_file.close()
            pred.clear()


        frame_counter += 1

    print("Keypoints extracted for all dyads successfully.")
    
