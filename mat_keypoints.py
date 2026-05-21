import numpy as np
import scipy.io as sio
import os
import json

# Folder directory
folder_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS/Processed Data 2/reprocess"
dst_dir_2d = "/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints"
dst_dir_3d = "/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/3D Keypoints"

# Define the empty frame (24 keypoints, 2D, 3D)
empty_frame_3d = np.zeros((24, 3))
empty_frame_2d = np.zeros((24, 2))

# Lists to hold keypoints over time
person_0_sequence = []
person_1_sequence = []
person_2_sequence = []

# Iterate over each frame
for folder in sorted(os.listdir(folder_dir)):

    print("Opening folder: " + folder)
    folder_path = os.path.join(folder_dir, folder)
    
    for file in sorted(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            print("Opening file " + file)
            dyad_info = json.load(f)
            f.close()

        frame_keypoints = {
            0: empty_frame_3d.copy(),
            1: empty_frame_3d.copy(),
            2: empty_frame_3d.copy()
        }

        for person in dyad_info.get("people", []):
            person_id = person.get("person_id")
            if person_id in frame_keypoints:
                keypoints = np.array(person["poses3d"]).reshape(24, 3)
                frame_keypoints[person_id] = keypoints
            else:
                print(f"Warning: Unexpected person_id {person_id} in {file}")

        # Add frame data to sequences
        person_0_sequence.append(frame_keypoints[0])
        person_1_sequence.append(frame_keypoints[1])
        person_2_sequence.append(frame_keypoints[2])
        
    person_0 = np.array(person_0_sequence)
    person_0 = np.transpose(person_0, (1, 2, 0))
    
    person_1 = np.array(person_1_sequence)
    person_1 = np.transpose(person_1, (1, 2, 0))
    
    person_2 = np.array(person_2_sequence)
    person_2 = np.transpose(person_2, (1, 2, 0))
    
    new_dict_3d = {"person_0_3d": person_0, "person_1_3d": person_1, "person_2_3d": person_2}
    # new_dict_3d = {"person_0_3d": person_0}
    new_file_name_3d = folder + ' 3D Keypoints.mat'
    print("Saving to " + os.path.join(dst_dir_3d, new_file_name_3d))
    sio.savemat(os.path.join(dst_dir_3d, new_file_name_3d), new_dict_3d)

    person_0_sequence.clear()
    person_1_sequence.clear()
    person_2_sequence.clear()
    



