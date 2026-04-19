import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from missing_gaps_stats import import_data
from signal_postprocessing import lin_interp_threshold, butterworth_filter, find_missing_segments_indices, movmad_filter

# Cleaning and interpolating signals for all dyads except excluded ones
GAP_THRESHOLD_FRAMES = 12
keypoints_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints"
excluded_dyads = [57, 76, 78, 112]
desired_joint_indices = [15, 16, 17]

def normalize_signal(data_x, data_y):
    normalized_signal = np.

def main():

    for file in os.listdir(keypoints_dir):
        keypoint_path = os.path.join(keypoints_dir, file)
        dyad_keypoints = import_data(keypoint_path)
        
        infant_keypoints = np.array(dyad_keypoints["Infant"])
        parent_keypoints = np.array(dyad_keypoints["Parent"])
        
        for joint in desired_joint_indices:
            for coordinate in range(2):
                infant_original_signal = infant_keypoints[joint, coordinate, :]
                parent_original_signal = parent_keypoints[joint, coordinate, :]

                # Use thresholded linear interpolation on missing data
                infant_interpolated_signal = lin_interp_threshold(infant_original_signal, GAP_THRESHOLD_FRAMES)
                parent_interpolated_signal = lin_interp_threshold(parent_original_signal, GAP_THRESHOLD_FRAMES)
                
                # Median filtering 
                infant_filtered_signal = movmad_filter(infant_interpolated_signal, 30)
                parent_filtered_signal = movmad_filter(parent_interpolated_signal, 30)
                
        # Normalize signal 
        infant_normalized_signal = np.sqrt((infant_keypoints[:, 0, :]**2) + (infant_keypoints[:, 1, :]**2))
        parent_normalized_signal = np.sqrt((parent_keypoints[:, 0, :]**2) + (parent_keypoints[:, 1, :]**2))
        
        
            
            
                
                
                
                                                   
                
                
                
                
                
                
                
                
                
