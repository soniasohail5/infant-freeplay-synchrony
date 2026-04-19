import numpy as np
from pathlib import Path
from pkg_resources import parse_version
import os
import tensorflow as tf
import json
import tensorflow_hub as hub
import cv2
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/infantresearch/tf219/tf219/lib/python3.12/site-packages/cv2/qt/plugins/platforms/libqxcb.so'

# Joint keypoint extraction from videos in real-time (frames are extracted from OpenCV and inputed to MeTRABS one at a time)
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
    image_ax.set_xlim(0, 800)
    image_ax.set_ylim(800, 0)
    image_ax.imshow(image.numpy())
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
        
folder_dir = "/mnt/e/IN-PERSON EXPERIMENT RECORDINGS/3HYPER FREEPLAY/3HYPER DV FREEPLAY/keypoint extraction"
dst_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS"
video_duration = 240

print("Loading model ....")
model = hub.load('https://bit.ly/metrabs_l')
skeleton = 'smpl_24'
joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

for file in os.listdir(folder_dir):
    cap = cv2.VideoCapture(os.path.join(folder_dir, file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    file_path = Path(file)

    print("Opening video file..." + file_path.stem)

    folder_name = file + " EXTRACTED"

    total_frames = int(fps) * video_duration
    frame_counter = 0
    ret = 1

    while ret or frame_counter < total_frames:

        ret, bgr_frame = cap.read()
        
        try:
            if ret == 1:
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) # Needed since OpenCV reads and saves images in BGR channels
                frame_input = tf.convert_to_tensor(rgb_frame)
                frame_counter_str = str(frame_counter)
                frame_name = "frame" + frame_counter_str.zfill(4) + " keypoints.json"
                image_name = "frame" + frame_counter_str.zfill(4) 
                new_file_path = os.path.join(dst_dir, file_path.stem, frame_name)
                new_img_path = os.path.join(dst_dir, file_path.stem, image_name)
                
                print("Running inference on " + image_name)
                pred = model.detect_poses(frame_input, default_fov_degrees=55, skeleton=skeleton, suppress_implausible_poses=True, detector_flip_aug=True, detector_nms_iou_threshold=0.4)
                pred = tf.nest.map_structure(lambda x: x.numpy(), pred)
                os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        
                visualize(frame_input, pred, joint_names, joint_edges, new_img_path)

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

        except cv2.error:
            continue