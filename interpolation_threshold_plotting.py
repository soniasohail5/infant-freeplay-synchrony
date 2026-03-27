import os
import numpy as np
import matplotlib.pyplot as plt
from missing_gaps_stats import import_data
from signal_postprocessing import replace_missing       
from signal_plotting import find_missing_segments_indices 

'''
This file analyzes the data loss incurred by applying thresholds on the maximum gap size and minimum number of known frames before and after a gap for developing an 
interpolation criteria that optimizes the tradeoff between retaining as much data as possible while ensuring that the interpolated values are accurate. 

The data loss is calculated on the 2D keypoint data extracted from the 3HYPER FREEPLAY DV videos using the MeTRABs model, which contains NaN values for missing frames. 

The keypoint data is in the format of a (joints, 2, frames) array where the last dimension corresponds to the x and y coordinates of each joint. 

The code iterates through different combinations of gap and known frame thresholds, applies linear interpolation to fill in missing values that meet the criteria, 
and calculates the percentage of data loss incurred for each joint under each threshold combination. 

3 materials prepared by this file 
- excel sheet w/ the data loss incurred by the gap size threshold from 10-30 frames per video (for simplicity, focus only on the head keypoint)
- figure that plots the data loss curve as a function of the gap size threshold and the known frame threshold (separate functions, focus only on the head keypoint, averaged across all videos)

'''
GAP_THRESHOLDS = np.arange(1, 31).tolist()     # max consecutive NaN frames allowed for interpolation
KNOWN_THRESHOLDS = np.arange(1, 31).tolist()    # minimum number of known frames before and after a gap for interpolation

def count_total_missing_frames(keypoint_data):
    missing_gaps = find_missing_segments_indices(keypoint_data)
    total_missing_frames = sum(len(gap) for gap in missing_gaps)
    
    return total_missing_frames

def count_known_frames_before_gap(keypoint_data, gap_start):
    count = 0
    for i in range(gap_start - 1, -1, -1):
        if not np.isnan(keypoint_data[i]):
            count += 1
        else:
            break
    return count

def count_known_frames_after_gap(keypoint_data, gap_end):
    count = 0
    for i in range(gap_end + 1, keypoint_data.shape[0]):
        if not np.isnan(keypoint_data[i]):
            count += 1
        else:
            break
    return count

def find_interpolated_gaps_by_known_frames(keypoint_data, known_frame_threshold):
    # Returns a list of gap segments that meet the known frame threshold criteria for interpolation
    missing_gaps = find_missing_segments_indices(keypoint_data)
    
    valid_gaps = []
    for gap in missing_gaps:
        start = gap[0]
        end = gap[-1]
        known_before = start - 1
        known_after = end + 1
        
        if known_before >= 0 and known_after < keypoint_data.shape[0]:
            valid_before = count_known_frames_before_gap(keypoint_data, start) >= known_frame_threshold
            valid_after = count_known_frames_after_gap(keypoint_data, end) >= known_frame_threshold
            
            if valid_before and valid_after:
                valid_gaps.append((start, end))
    
    return valid_gaps

def find_interpolated_gaps_by_gap_size(keypoint_data, gap_size_threshold):
    # Returns a list of gap segments that meet the gap size threshold criteria for interpolation
    missing_gaps = find_missing_segments_indices(keypoint_data)

    valid_gaps = []
    for gap in missing_gaps:
        start = gap[0]
        end = gap[-1]
        if len(gap) <= gap_size_threshold:
            valid_gaps.append((start, end))
    
    return valid_gaps

def calculate_joint_data_loss(keypoint_data, gap_threshold, known_threshold):
    # Return the percentage of data loss incurred for a given joint under the specified interpolation thresholds
    if known_threshold != None and gap_threshold != None:
        return None, None # only one parameter is tested at a time 
    if known_threshold == None:
        valid_gaps = find_interpolated_gaps_by_gap_size(keypoint_data, gap_threshold)
    if gap_threshold == None:
        valid_gaps = find_interpolated_gaps_by_known_frames(keypoint_data, known_threshold)
    
    total_selected_frames = 0
    for gap in valid_gaps:
        gap_size = len(gap)
        total_selected_frames += gap_size
        
    total_missing_frames = count_total_missing_frames(keypoint_data)
    total_frames = keypoint_data.shape[0]
    
    total_rejected_frames = total_missing_frames - total_selected_frames
    if total_frames == 0:
        return 0, 0
    data_loss_percentage = (total_rejected_frames / total_frames) * 100


def calculate_gap_recovery(keypoint_data, gap_threshold, known_threshold):
    # Return the percentage of missing frames that are successfully recovered (i.e. interpolated) under the specified interpolation thresholds
    if known_threshold != None and gap_threshold != None:
        return None, None # only one parameter is tested at a time 
    if known_threshold == None:
        valid_gaps = find_interpolated_gaps_by_gap_size(keypoint_data, gap_threshold)
    if gap_threshold == None:
        valid_gaps = find_interpolated_gaps_by_known_frames(keypoint_data, known_threshold)
    
    total_selected_frames = 0
    for gap in valid_gaps:
        gap_size = len(gap)
        total_selected_frames += gap_size
        
    total_missing_frames = count_total_missing_frames(keypoint_data)
    
    if total_missing_frames == 0:
        return 0, 0
    recovery_percentage = (total_selected_frames / total_missing_frames) * 100
    
    return total_selected_frames, recovery_percentage

def plot_data_loss_curves(infant_gap_loss_percentages, infant_known_loss_percentages, parent_gap_loss_percentages, parent_known_loss_percentages):
    # Plot the data loss curves as a function of the gap size threshold and the known frame threshold
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(GAP_THRESHOLDS, infant_gap_loss_percentages, marker='o', color='blue')
    plt.plot(KNOWN_THRESHOLDS, infant_known_loss_percentages, marker='o', color='orange')
    plt.title('Data Loss - Infant')
    plt.xlabel('Threshold (frames)')
    plt.ylabel('Data Loss (%)')

    plt.subplot(1, 2, 2)
    plt.plot(GAP_THRESHOLDS, parent_gap_loss_percentages, marker='o', color='blue')
    plt.plot(KNOWN_THRESHOLDS, parent_known_loss_percentages, marker='o', color='orange')
    plt.title('Data Loss - Parent')
    plt.xlabel('Threshold (frames)')
    plt.ylabel('Data Loss (%)')

    plt.tight_layout()
    plt.show()

def main():
    # Load your keypoint data
    folder_path = "/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints/"
    dyad_number = int(input("Enter dyad number: "))
    if dyad_number >= 100:
        file_path = folder_path + "3HYPER." + str(dyad_number) + " FREEPLAY DV EXTRACTED 2D Keypoints.mat"
    elif dyad_number >= 10:
        file_path = folder_path + "3HYPER.0" + str(dyad_number) + " FREEPLAY DV EXTRACTED 2D Keypoints.mat"
    else:
        file_path = folder_path + "3HYPER.00" + str(dyad_number) + " FREEPLAY DV EXTRACTED 2D Keypoints.mat"
        
    dyad_keypoint_data = import_data(file_path)
    infant_keypoint_data, infant_nan = replace_missing(dyad_keypoint_data["infant"])
    parent_keypoint_data, parent_nan = replace_missing(dyad_keypoint_data["parent"])
    
    infant_keypoint_data = infant_keypoint_data[15, 0, :] # focus on head keypoint for simplicity
    parent_keypoint_data = parent_keypoint_data[15, 0, :] # focus on head keypoint for simplicity
    
    infant_gap_loss_percentages = []
    infant_known_loss_percentages = []
    parent_gap_loss_percentages = []
    parent_known_loss_percentages = []
    
    for gap_threshold in GAP_THRESHOLDS:
        infant_total_selected_frames, infant_data_loss_percentage = calculate_joint_data_loss(infant_keypoint_data, gap_threshold, None)
        parent_total_selected_frames, parent_data_loss_percentage = calculate_joint_data_loss(parent_keypoint_data, gap_threshold, None)
        infant_gap_loss_percentages.append(infant_data_loss_percentage)
        parent_gap_loss_percentages.append(parent_data_loss_percentage)
        print(f"Infant - Gap Threshold: {gap_threshold} frames - Total Interpolated Frames: {infant_total_selected_frames}, Data Loss: {infant_data_loss_percentage:.2f}%")
        print(f"Parent - Gap Threshold: {gap_threshold} frames - Total Interpolated Frames: {parent_total_selected_frames}, Data Loss: {parent_data_loss_percentage:.2f}%")
    
    for known_threshold in KNOWN_THRESHOLDS:
        infant_total_selected_frames, infant_data_loss_percentage = calculate_joint_data_loss(infant_keypoint_data, None, known_threshold)
        parent_total_selected_frames, parent_data_loss_percentage = calculate_joint_data_loss(parent_keypoint_data, None, known_threshold)
        infant_known_loss_percentages.append(infant_data_loss_percentage)
        parent_known_loss_percentages.append(parent_data_loss_percentage)
        print(f"Infant - Known Frame Threshold: {known_threshold} frames - Total Interpolated Frames: {infant_total_selected_frames}, Data Loss: {infant_data_loss_percentage:.2f}%")
        print(f"Parent - Known Frame Threshold: {known_threshold} frames - Total Interpolated Frames: {parent_total_selected_frames}, Data Loss: {parent_data_loss_percentage:.2f}%")
        
    plot_data_loss_curves(infant_gap_loss_percentages, infant_known_loss_percentages, parent_gap_loss_percentages, parent_known_loss_percentages)

if __name__ == "__main__":
    main()