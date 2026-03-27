#%%
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import tensorflow_hub as hub

#%%

folder_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS/3HYPER.0"

# Load METRABS model (only for skeleton and joint edges/connections)
joint_names = np.loadtxt("/mnt/c/metrabs joint connections/joint_names.txt", dtype='str')
joint_edges = np.loadtxt("/mnt/c/metrabs joint connections/joint_edges.txt", dtype='int')

# Label mapping
person_ids = [0, 1, 2]  # Infant, parent, or other (object or other individual)

# Prompt user to select the subject number for visualization
subject_num = int(input("Enter Subject ID: "))

# Create full folder path (frame)
if subject_num >= 10:
    folder_path = folder_dir + str(subject_num) + " FREEPLAY DV EXTRACTED"
else:
    folder_path = folder_dir + '0' + str(subject_num) + " FREEPLAY DV EXTRACTED"

# Initialize variables to store min/max x and y coordinates
x_max = 0
x_min = 800
y_max = 0
y_min = 800

# Accesses the folder with the chosen subject's keypoints. Error is thrown if the subject doesn't exist in the directory

if not os.path.exists(folder_path):
    print("Subject ID not found.")
else:
    # Number of frames for each subject
    frame_counter = 0

    # Opens JSON file with detected keypoints for each frame
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            fig = plt.figure(figsize=(10, 5.2))
            image_ax = fig.add_subplot(1, 2, 1)

            pose_ax = fig.add_subplot(1, 2, 2, projection='3d')
            pose_ax.view_init(5, -85)

            pose_ax.set_xlim3d(-1500, 1500)
            pose_ax.set_zlim3d(-1500, 1500)
            pose_ax.set_ylim3d(0, 3000)
            image_ax.set_xlim(-1000, 1500)
            image_ax.set_ylim(-1000, 1500)

            pose_ax.set_box_aspect((1, 1, 1))
            open_file = open(os.path.join(folder_path, file), "r")
            dyad_info = json.load(open_file)

            for person in dyad_info["people"]:
                person_id = person["person_id"]
                print(person_id)
                poses3d = np.array(person["poses3d"])
                poses2d = np.array(person["poses2d"])
                poses3d = poses3d.reshape(24, 3)
                poses2d = poses2d.reshape(24, 2)

                x_max_temp = poses3d[:, 0].max()
                x_min_temp = poses3d[:, 0].min()
                y_max_temp = poses3d[:, 1].max()
                y_min_temp = poses3d[:, 1].min()
                z_min_temp = poses3d[:, 2].min()
                z_max_temp = poses3d[:, 2].max()

                if x_max_temp > x_max:
                    x_max = x_max_temp
                if x_min_temp < x_min:
                    x_min = x_min_temp
                if y_max_temp > y_max:
                    y_max = y_max_temp
                if y_min_temp < y_min:
                    y_min = y_min_temp

                if person_id[0] == 0:
                    color = 'r'
                elif person_id[0] == 1:
                    color = 'b'
                else:
                    color = 'g'

                # Matplotlib plots the Z axis as vertical, but our poses have Y as the vertical axis.
                # Therefore, we do a 90° rotation around the X axis:
                poses3d[..., 1], poses3d[..., 2] = poses3d[..., 2], -poses3d[..., 1]
                for i_start, i_end in joint_edges:
                    print(i_start, i_end)
                    joint2d_x = [poses2d[i_start, 0], poses2d[i_end, 0]]
                    joint2d_y = [poses2d[i_start, 1], poses2d[i_end, 1]]

                    joint3d_x = [poses3d[i_start, 0], poses3d[i_end, 0]]
                    joint3d_y = [poses3d[i_start, 2], poses3d[i_end, 2]]
                    joint3d_z = [poses3d[i_start, 1], poses3d[i_end, 1]]

                    image_ax.plot(joint2d_x, joint2d_y, linestyle='solid', color=color, marker='o', markersize=4)
                    pose_ax.plot3D(joint3d_x, joint3d_y, joint3d_z, linestyle='solid', marker='o', markersize=4, color=color)

                image_ax.scatter(poses2d[:, 0], poses2d[:, 1], s=2, color=color)
                pose_ax.scatter(poses3d[:, 0], poses3d[:, 2], poses3d[:, 1], color=color)

            fig.tight_layout()
            image_ax.invert_yaxis()
            plt.show()
            plt.pause(0.033)
            plt.close(fig)


        frame_counter += 1
        open_file.close()

# %%