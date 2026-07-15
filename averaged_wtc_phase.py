import os
import numpy as np
import matplotlib.pyplot as plt
from single_dyad_wtc_analysis import load_data_mat, compute_wtc, DESIRED_JOINT_NAMES
from numpy.lib.stride_tricks import sliding_window_view

'''
Windowed averaging of wavelet coherence and phase angles computed from head and shoulder keypoint data for a dyad within a specific frequency band.

FOI: 0.5-2Hz (similar to the range mentioned in the 2022 Fujiwara et al. paper)
Window size: 10s + 2s overlap
Specified keypoints: Head, L/R shoulders (averaged)

'''
WINDOW_SIZE_SECONDS = 10 
WINDOW_OVERLAP_SECONDS = 2
SELECTED_JOINT_NAMES = ['Head', 'Left Shoulder', 'Right Shoulder']
JOINT_MOVEMENT_PATH = "/mnt/c/3HYPER FREEPLAY DV METRABs/MATLAB Keypoints 2/2D Keypoints Processed/3HYPER.025 FREEPLAY DV PROCESSED 2D Keypoints.mat"

def convert_seconds_to_frame(seconds, frame_rate):
    return int(seconds * frame_rate)

def make_sliding_windows(data, window_size_frames, overlap):
    step = window_size_frames - overlap
    num_windows = (data.shape[1] - window_size_frames) // step + 1
    windows = sliding_window_view(data, window_shape=(window_size_frames, data.shape[1]))[::step]
    return windows[:num_windows]

def calculate_average_joint_movement(joint_data, selected_joint_names):
    joint_indices = [DESIRED_JOINT_NAMES.index(name) for name in selected_joint_names]
    selected_joint_data = joint_data[:, joint_indices, :]
    average_joint_movement = np.mean(selected_joint_data, axis=1)
    return average_joint_movement

def main():
    # Load movement data 
    dyad_info = load_data_mat(JOINT_MOVEMENT_PATH)
    frame_rate = 27.49 # taken from database, but needs to be adjusted for every sample
    
    # Convert window size and overlap from seconds to frames
    window_size_frames = convert_seconds_to_frame(WINDOW_SIZE_SECONDS, frame_rate)
    overlap_frames = convert_seconds_to_frame(WINDOW_OVERLAP_SECONDS, frame_rate)
    
    # Calculate average joint movement for selected joints
    average_joint_movement = calculate_average_joint_movement(dyad_info['joint_data'], SELECTED_JOINT_NAMES[1:]) # exclude head 
    
    # Create sliding windows for the average joint movement
    windows = make_sliding_windows(average_joint_movement, window_size_frames, overlap_frames)
    
    # Compute wavelet coherence and phase angles for entire signal, average across FOI, then average across windows
    head_wtc_signal, head_phase_signal, head_coi, head_freqs, hsig = compute_wtc(dyad_info, SELECTED_JOINT_NAMES[0]) # head vs shoulders
    


