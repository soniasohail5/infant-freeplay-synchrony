import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from missing_gaps_stats import import_data, get_dyad_number
from signal_postprocessing import replace_missing, lin_interp_threshold, movmad_filter

# Cleaning and interpolating signals for all dyads except excluded ones
GAP_THRESHOLD_FRAMES = 12
JOINT_NAMES = ["Neck", "Head", "Left Shoulder", "Right Shoulder"]
keypoints_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints"
dst_dir = "/mnt/c/3HYPER FREEPLAY DV METRABs/MATLAB Keypoints 2/2D Keypoints Processed"
excluded_dyads = [57, 76, 78, 112, 40, 108, 31]
desired_joint_indices = [12, 15, 16, 17, 18, 19]

def save_keypoints_in_mat(dyad_number, processed_infant_data, processed_parent_data, destination_folder_path):
    # Prepare file path for saving
    if dyad_number >= 100:
        file_name = "3HYPER." + str(dyad_number) + " FREEPLAY DV PROCESSED 2D Keypoints.mat"
    else:
        file_name = "3HYPER.0" + str(dyad_number) + " FREEPLAY DV PROCESSED 2D Keypoints.mat"
            
    full_file_path = os.path.join(destination_folder_path, file_name)
        
    # Create dictionary object to place infant and parent keypoints 
    dyad_keypoints = {"Infant": processed_infant_data, "Parent": processed_parent_data}
    print(f"Saving to .... {full_file_path}")
    sio.savemat(full_file_path, dyad_keypoints)

def normalize_signal(data_x, data_y):
    normalized_signal = np.sqrt(np.pow(data_x, 2) + np.pow(data_y, 2))
    
    return normalized_signal

def plot_original_vs_preprocessed_signals(infant_original_data, parent_original_data, infant_normalized_data, parent_normalized_data, dyad_number, joint_index):
    fig, ax = plt.subplots(2, 2, sharex=True)
    
    frames_x = np.arange(len(parent_original_data))
    
    fig.suptitle(f"Original vs. Preprocessed for Dyad #{dyad_number}")

    ax[0,0].plot(frames_x, infant_original_data)
    ax[0,0].set_title(f"Infant (Original)")
    
    ax[1,0].plot(frames_x, infant_normalized_data)
    ax[1,0].set_title(f"Infant (Preprocessed)")
    
    ax[0,1].plot(frames_x, parent_original_data)
    ax[0,1].set_title(f"Parent (Original)")
    
    ax[1,1].plot(frames_x, parent_normalized_data)
    ax[1,1].set_title(f"Parent (Preprocessed)")
    
    plt.show()

def main():

    for file in os.listdir(keypoints_dir):
        keypoint_path = os.path.join(keypoints_dir, file)
        dyad_number = get_dyad_number(file)
        
        if dyad_number in excluded_dyads:
            print(f"Skipping Dyad #{dyad_number} due to exclusion criteria.")
            continue
        
        print(f"Loading Dyad #{dyad_number} .....")
        dyad_keypoints = import_data(keypoint_path)
        infant_keypoints = np.array(dyad_keypoints["infant"])
        parent_keypoints = np.array(dyad_keypoints["parent"])
        
        infant_max_frames = infant_keypoints.shape[2]
        parent_max_frames = parent_keypoints.shape[2]
        
        infant_intermediate_keypoints = np.ones_like(infant_keypoints)
        parent_intermediate_keypoints = np.ones_like(parent_keypoints)
        
        infant_modified_keypoints = np.zeros((len(desired_joint_indices), infant_max_frames))
        parent_modified_keypoints = np.zeros((len(desired_joint_indices), parent_max_frames))
        
        for (joint, new_joint) in (zip(desired_joint_indices, range(len(desired_joint_indices)))):
            for coordinate in range(2):
                infant_original_signal = infant_keypoints[joint, coordinate, :]
                parent_original_signal = parent_keypoints[joint, coordinate, :]
                
                # Replace 0s with NaN
                infant_original_signal, _ = replace_missing(infant_original_signal)
                parent_original_signal, _ = replace_missing(parent_original_signal)

                # Use thresholded linear interpolation on missing data
                print(f"Interpolating keypoints .....")
                infant_interpolated_signal = lin_interp_threshold(infant_original_signal, GAP_THRESHOLD_FRAMES)
                parent_interpolated_signal = lin_interp_threshold(parent_original_signal, GAP_THRESHOLD_FRAMES)
                
                # Median filtering 
                print("Filtering keypoints with median filtering .....")
                infant_intermediate_keypoints[new_joint, coordinate, :] = movmad_filter(infant_interpolated_signal, 30)
                parent_intermediate_keypoints[new_joint, coordinate, :] = movmad_filter(parent_interpolated_signal, 30)
                
            # Normalize signal 
            print("Normalize signal using L2 norm .....")
            infant_x, infant_y = infant_intermediate_keypoints[new_joint, 0, :], infant_intermediate_keypoints[new_joint, 1, :]
            parent_x, parent_y = parent_intermediate_keypoints[new_joint, 0, :], parent_intermediate_keypoints[new_joint, 1, :]
            
            infant_normalized_signal = normalize_signal(infant_x, infant_y)
            parent_normalized_signal = normalize_signal(parent_x, parent_y)
            
            # Add nornalized keypoints to new numpy array for saving
            print(f"Saving modified signals to new array ......")
            infant_modified_keypoints[new_joint, :] = infant_normalized_signal
            parent_modified_keypoints[new_joint, :] = parent_normalized_signal
            
            # Plot keypoints for verification
            plot_original_vs_preprocessed_signals(infant_keypoints[joint, coordinate, :], parent_keypoints[joint, coordinate, :], infant_modified_keypoints[new_joint, :], 
                                                  parent_modified_keypoints[new_joint, :], dyad_number)
        
        # Save keypoints to .mat file for analysis 
        save_keypoints_in_mat(dyad_number, infant_modified_keypoints, parent_modified_keypoints, dst_dir)
            
if __name__ == "__main__":
    main()

            
            
            
            
            
        
        
            
            
                
                
                
                                                   
                
                
                
                
                
                
                
                
                
