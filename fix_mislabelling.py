import os
import copy
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from signal_postprocessing import replace_missing
from itertools import chain
from missing_gaps_stats import import_data
from signal_postprocessing import lin_interp_threshold

keypoints_folder = '/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints'
dyad_mislabelling_list = '3HYPER Joint Keypoint Label Swapping Log.csv'
dst_dir = '/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints Swapped'
selected_joint_indices = [16]

def import_unknown_detection_keypoints(keypoints_path):
    # Extracts person_2 keypoints for swapping purposes (identified as unknown in swapping log)
    dyad_info = sio.loadmat(keypoints_path)
    unknown_keypoints = np.array(dyad_info["person_2_2d"])
    print(type(unknown_keypoints))
    print(unknown_keypoints.shape[2])
    
    return unknown_keypoints

def filter_nans_from_indices(infant_signal, parent_signal, selected_indices):
    infant_signal, _ = replace_missing(infant_signal)
    parent_signal, _ = replace_missing(parent_signal)

    all_selected_indices = np.array(list(chain.from_iterable(selected_indices)), dtype=int)

    valid_mask = (
        ~np.isnan(infant_signal[all_selected_indices]) &
        ~np.isnan(parent_signal[all_selected_indices])
    )
    
    filtered_indices = all_selected_indices[valid_mask].tolist()
    
    nan_indices = np.where(~np.isnan(infant_signal[all_selected_indices]) & ~np.isnan(parent_signal[all_selected_indices]))
    
    for index in nan_indices:
        nearest_point = filtered_indices
    return filtered_indices

def import_select_participants(list_of_unique_dyads, folder_path):
    dyad_info_collection = {}
    
    for dyad in list_of_unique_dyads:
        
        if dyad >= 100:
            file_name = "3HYPER." + str(dyad) + " FREEPLAY DV EXTRACTED 2D Keypoints.mat"
        else:
            file_name = "3HYPER.0" + str(dyad) + " FREEPLAY DV EXTRACTED 2D Keypoints.mat"
            
        file_path = os.path.join(folder_path, file_name)
        
        dyad_keypoints = import_data(file_path)
        
        dyad_info_collection[dyad] = dyad_keypoints
        
    return dyad_info_collection

def swap_keypoints(infant_data, parent_data, list_of_indices):
    original_infant = copy.deepcopy(infant_data)
    swapped_infant = copy.deepcopy(infant_data)
    original_parent = copy.deepcopy(parent_data)
    swapped_parent = copy.deepcopy(parent_data)
    
    list_of_indices = list(chain.from_iterable(list_of_indices))
    swapped_infant[list_of_indices] = original_parent[list_of_indices]
    swapped_parent[list_of_indices] = original_infant[list_of_indices]
        
    return swapped_infant, swapped_parent

def find_all_swap_indices(swap_log_df, dyad_number):
    dyad_swap_log_df = swap_log_df[swap_log_df['Dyad Number'] == dyad_number]
    dyad_swap_log_df_filtered = dyad_swap_log_df[swap_log_df.loc[dyad_swap_log_df.index, 'Notes'].fillna('').astype(str).str.strip() == ''] # only filter sections with no additional notes 
    
    list_swap_indices = []
    
    for start, end in zip(dyad_swap_log_df_filtered['Start Frame Number'], dyad_swap_log_df_filtered['End Frame Number']):
        frames = np.arange(start, end)
        list_swap_indices.append(frames)
        
    return list_swap_indices

def save_keypoints_in_mat(dyad_number, swapped_infant_data, swapped_parent_data, destination_folder_path):
    # Prepare file path for saving
    if dyad_number >= 100:
        file_name = "3HYPER." + str(dyad_number) + " FREEPLAY DV EXTRACTED 2D Swapped Keypoints.mat"
    else:
        file_name = "3HYPER.0" +str(dyad_number) + " FREEPLAY DV EXTRACTED 2D Swapped Keypoints.mat"
            
    full_file_path = os.path.join(destination_folder_path, file_name)
        
    # Create dictionary object to place infant and parent keypoints 
    dyad_keypoints = {"Infant": swapped_infant_data, "Parent": swapped_parent_data}
    print(f"Saving to .... {full_file_path}")
    sio.savemat(full_file_path, dyad_keypoints)
    
def plot_interval(original_data, swapped_data, joint, coordinate, start=0, end=7000, title="Original vs Swapped"):
    original_signal = original_data[start:end]
    swapped_signal = swapped_data[start:end]

    t = np.arange(start, end)

    plt.figure(figsize=(12, 5))
    plt.plot(t, original_signal, label="Original", alpha=0.7)
    plt.plot(t, swapped_signal + 10, label="Swapped", alpha=0.7)

    plt.title(f"{title} | Joint {joint}, Coord {coordinate} | Frames {start}-{end}")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
        
def main():
    
    selected_dyads = [40]
    selected_dyad_keypoints = import_select_participants(selected_dyads, keypoints_folder)
    swap_log = pd.read_csv(dyad_mislabelling_list)
    
    for dyad in selected_dyad_keypoints:
        original_infant_selected_dyad_keypoints = copy.deepcopy(selected_dyad_keypoints[dyad]["infant"])
        original_parent_selected_dyad_keypoints = copy.deepcopy(selected_dyad_keypoints[dyad]["parent"])
        
        print(f"Swapping keypoints for Dyad #{dyad}")
        all_swap_indices = find_all_swap_indices(swap_log, dyad)
        print(f"Swapping at Indices: {all_swap_indices}")
        
        for joint in selected_joint_indices:
            for coordinate in range(2):
                # Get original keypoint data
                infant_signal, _ = replace_missing(lin_interp_threshold(selected_dyad_keypoints[dyad]["infant"][joint, coordinate, :], 12))
                parent_signal, _ = replace_missing(lin_interp_threshold(selected_dyad_keypoints[dyad]["parent"][joint, coordinate, :], 12))
                
                # Swap keypoints at necessary indices
                swapped_infant_signal, swapped_parent_signal = swap_keypoints(infant_signal, parent_signal, all_swap_indices)
                swapped_infant_signal, _ = replace_missing(swapped_infant_signal)
                swapped_parent_signal, _ = replace_missing(swapped_parent_signal)
                print("Swapping complete.")
                
                start=0
                stop=len(swapped_infant_signal)
                
                print(f"Original infant: {infant_signal[5190:5202]}")
                print(f"Swapped infant: {swapped_infant_signal[5190:5202]}")
                print(f"Original parent: {parent_signal[5190:5202]}")
                print(f"Swapped parent: {swapped_parent_signal[5190:5202]}")
                
                selected_dyad_keypoints[dyad]["infant"][joint, coordinate, :] = swapped_infant_signal
                selected_dyad_keypoints[dyad]["parent"][joint, coordinate, :] = swapped_parent_signal
                
                # plot_interval(infant_signal, swapped_infant_signal, joint, coordinate, start, stop)
                
        # Save keypoints to new .mat file
        # print(f"Saving swapped keypoints for Dyad #{dyad}")
        # save_keypoints_in_mat(dyad, selected_dyad_keypoints[dyad]["infant"], selected_dyad_keypoints[dyad]["parent"], dst_dir)
        
if __name__ == "__main__":
    main()
                
        
        
                
                
                
                

                
                
                
    
                
        
                
   
                
        
        
        
        
    
    
    