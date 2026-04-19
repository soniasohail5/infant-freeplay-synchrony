import os
import copy
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from signal_postprocessing import replace_missing
from missing_gaps_stats import import_data, find_missing_segments_indices

keypoints_folder = '/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints'
dyad_mislabelling_list = '3HYPER Joint Keypoint Label Swapping Log.csv'
dst_dir = '/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints Swapped'
selected_joint_indices = [15, 16, 17]

def filter_nans_from_indices(infant_data, parent_data, selected_indices):
    filtered_infant_data, infant_nan = replace_missing(infant_data)
    filtered_parent_data, parent_nan = replace_missing(parent_data)
    
    for indices in selected_indices:
        start = indices[0]
        end = indices[-1]
        if np.isnan(filtered_infant_data[:, :, start:end]).any() or np.isnan(filtered_parent_data[:, :, start:end]).any():
            selected_indices.remove(indices)
            
    return selected_indices

def import_select_participants(list_of_unique_dyads, folder_path):
    dyad_info_collection = {}
    
    for dyad in list_of_unique_dyads:
        
        if dyad >= 100:
            file_name = "3HYPER." + str(dyad) + " FREEPLAY DV EXTRACTED 2D Keypoints.mat"
        else:
            file_name = "3HYPER.0" +str(dyad) + " FREEPLAY DV EXTRACTED 2D Keypoints.mat"
            
        file_path = os.path.join(folder_path, file_name)
        
        dyad_keypoints = import_data(file_path)
        
        dyad_info_collection[dyad] = dyad_keypoints
        
    return dyad_info_collection

def swap_keypoints(infant_data, parent_data, list_of_indices):
    original_infant = copy.deepcopy(infant_data)
    swapped_infant = copy.deepcopy(infant_data)
    original_parent = copy.deepcopy(parent_data)
    swapped_parent = copy.deepcopy(parent_data)
    
    for indices in list_of_indices:
        start = indices[0]
        end = indices[-1]
        swapped_infant[start:end] = original_parent[start:end]
        swapped_parent[start:end] = original_infant[start:end]
        
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
        
def main():
    
    selected_dyads = [40, 59, 108]
    selected_dyad_keypoints = import_select_participants(selected_dyads, keypoints_folder)
    swap_log = pd.read_csv(dyad_mislabelling_list)
    
    for dyad in selected_dyad_keypoints:
        original_infant_selected_dyad_keypoints = copy.deepcopy(selected_dyad_keypoints[dyad]["infant"])
        original_parent_selected_dyad_keypoints = copy.deepcopy(selected_dyad_keypoints[dyad]["parent"])
        
        print(f"Swapping keypoints for Dyad #{dyad}")
        all_swap_indices = find_all_swap_indices(swap_log, dyad)
        filtered_swapped_indices = filter_nans_from_indices(original_infant_selected_dyad_keypoints, original_parent_selected_dyad_keypoints, all_swap_indices)
        
        for joint in selected_joint_indices:
            for coordinate in range(2):
                # Get original keypoint data
                infant_signal = selected_dyad_keypoints[dyad]["infant"][joint, coordinate, :]
                parent_signal = selected_dyad_keypoints[dyad]["parent"][joint, coordinate, :]
                
                # Swap keypoints at necessary indices
                swapped_infant_signal, swapped_parent_signal  = swap_keypoints(infant_signal, parent_signal, filtered_swapped_indices)
                print("Swapping complete.")
                
                selected_dyad_keypoints[dyad]["infant"][joint, coordinate, :] = swapped_infant_signal
                selected_dyad_keypoints[dyad]["parent"][joint, coordinate, :] = swapped_parent_signal
                
        # Save keypoints to new .mat file
        print(f"Saving swapped keypoints for Dyad #{dyad}")
        save_keypoints_in_mat(dyad, selected_dyad_keypoints[dyad]["infant"], selected_dyad_keypoints[dyad]["parent"], dst_dir)
        
if __name__ == "__main__":
    main()
                
        
        
                
                
                
                

                
                
                
    
                
        
                
   
                
        
        
        
        
    
    
    