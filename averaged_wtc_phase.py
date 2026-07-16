import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean
from single_dyad_wtc_analysis import load_data_mat, compute_wtc, DESIRED_JOINT_NAMES
from numpy.lib.stride_tricks import sliding_window_view

'''
Windowed averaging of wavelet coherence and phase angles computed from head and shoulder keypoint data for a dyad within a specific frequency band.

FOI: 0.5-2Hz (similar to the range mentioned in the 2022 Fujiwara et. al paper)
Window size: 10s + 2s overlap
Specified keypoints: Head, L/R shoulders (averaged)

7/16/2026 Note: To determine the dominant phase relationship within the dyad, phase angles will be binned into 4 groups ranging within 45 degrees each (in-phase, anti-phase, infant-led, parent-led)
A histogram will be plotted to determine the most frequent phase relationship
Average phase angles will be calculated using circular mean to avoid standard arithmetic

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

def plot_average_wtc_phase(avg_wtc_windows, avg_phase_windows, window_size_seconds, overlap_seconds):
    # add legend for head and shoulder joints
    num_windows = avg_wtc_windows.shape[0]
    time_axis = np.arange(num_windows) * (window_size_seconds - overlap_seconds)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, avg_wtc_windows, label='Average WTC', color='blue')
    plt.title('Average Wavelet Coherence (WTC) Across Windows')
    plt.xlabel('Time (s)')
    plt.ylabel('WTC')
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, avg_phase_windows, label='Average Phase Angle', color='orange')
    plt.title('Average Phase Angle Across Windows')
    plt.xlabel('Time (s)')
    plt.ylabel('Phase Angle (radians)')
    plt.grid()
    
    plt.tight_layout()
    plt.show()

def main():
    # Load movement data 
    dyad_info = load_data_mat(JOINT_MOVEMENT_PATH)
    frame_rate = 27.49 # taken from database, but needs to be adjusted for every sample
    max_frames = len(dyad_info["Parent"]["Head"])
    dt = 1/frame_rate # period of the signal (s)
    s0 = 2 * dt  # smallest scale of the wavelet transform
    
    # Convert window size and overlap from seconds to frames
    window_size_frames = convert_seconds_to_frame(WINDOW_SIZE_SECONDS, frame_rate)
    overlap_frames = convert_seconds_to_frame(WINDOW_OVERLAP_SECONDS, frame_rate)
    
    # Calculate average joint movement for selected joints and add them back to dyad_info 
    dyad_info["Infant"]["Shoulders Averaged"] = calculate_average_joint_movement(dyad_info["Infant"], SELECTED_JOINT_NAMES[1:]) 
    dyad_info["Parent"]["Shoulders Averaged"] = calculate_average_joint_movement(dyad_info["Parent"], SELECTED_JOINT_NAMES[1:])
    
    # Compute wavelet coherence and phase angles for entire signal, average across FOI, then average across windows
    #  WTC and phase angles for the head and shoulders
    head_wtc_signal, head_phase_signal, head_coi, head_freqs, hsig = compute_wtc(dyad_info, SELECTED_JOINT_NAMES[0]) 
    shoulder_wtc_signal, shoulder_phase_signal, shoulder_coi, shoulder_freqs, ssig = compute_wtc(dyad_info, "Shoulders Averaged") 
    
    # Extract data from FOI 
    foi_indices = np.where((head_freqs >= 0.5) & (head_freqs <= 2))[0]
    head_wtc_foi = head_wtc_signal[foi_indices, :]
    head_phase_foi = head_phase_signal[foi_indices, :]
    shoulder_wtc_foi = shoulder_wtc_signal[foi_indices, :]
    shoulder_phase_foi = shoulder_phase_signal[foi_indices, :]
    
    # Bin phase values to determine dominant phase relationship
    phase_labels = ["In-Phase", "Anti-Phase", "Infant-Leading", "Parent-Leading"]
    head_phase_foi_dg, shoulder_phase_foi_dg = np.degrees(head_phase_foi) % 360, np.degrees(shoulder_phase_foi) % 360
    
    head_phase_binned = {}
    shoulder_phase_binned = {} 
    
    head_in_phase = np.sum((head_phase_foi_dg <= 45) | (head_phase_foi_dg > 315))
    head_parent_lead = np.sum((head_phase_foi_dg > 45) | (head_phase_foi_dg <= 135))
    head_infant_lead = np.sum((head_phase_foi_dg > 135) | (head_phase_foi_dg <= 225))
    head_parent_lead = np.sum((head_phase_foi_dg > 225) | (head_phase_foi_dg <= 315)) 
    
    # Average the WTC and phase angles across the sliding windows
    avg_head_wtc_windowed = np.mean(make_sliding_windows(head_wtc_foi, window_size_frames, overlap_frames), axis=1)
    avg_shoulder_wtc_windowed = np.mean(make_sliding_windows(shoulder_wtc_foi, window_size_frames, overlap_frames), axis=1)
    avg_head_phase_windowed = circmean(make_sliding_windows(head_phase_foi, window_size_frames, overlap_frames))
    avg_shoulder_phase_windowed = circmean(make_sliding_windows(shoulder_phase_foi, window_size_frames, overlap_frames))
    
    # Plot the averaged WTC and phase angles across windows
    



