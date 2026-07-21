import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean
from itertools import product
from single_dyad_wtc_analysis import load_data_mat, compute_wtc
from numpy.lib.stride_tricks import sliding_window_view

'''
Windowed averaging of wavelet coherence and phase angles computed from head and shoulder keypoint data for a dyad within a specific frequency band.

FOI: 0.5-2Hz (similar to the range mentioned in the 2022 Fujiwara et. al paper)
Window size: 10s + 2s overlap
Specified keypoints: Head, L/R shoulders (averaged)

7/16/2026 Note: To determine the dominant phase relationship within the dyad, phase angles will be binned into 4 groups ranging within 45 degrees each (in-phase, anti-phase, infant-led, parent-led)
A histogram will be plotted to determine the most frequent phase relationship
Average phase angles will be calculated using circular mean to avoid errors with standard arithmetic

7/19/2026 Note: Another method of calculating average shoulder coherence will be tested
Rather than averaging shoulder movement and using it as input for WTC, the WTC from right and left shoulders will be computed separately 
and then averaged to obtain the shoulder coherence. This is done to minimize the loss of phase information that occurs 
when the shoulder movements are averaged prior to computing WTC.

7/21/2026 Note: The question has shifted to determining which pair of shoulder joints yields significant changes in coherence within the dyad. 
All possible shoulder combinations will be analyzed to determine which pair yields significant changes in coherence within the dyad.
This will also be extended to the elbow keypoints to determine which pair of elbow joints yields significant changes in coherence within the dyad.

Modifications:
- Separate figures for averaged WTC and averaged phase angle for shoulders and joints
    (4 figures, phase angle shoulders, phase angle elbows, WTC shoulders, WTC elbows)
- Each figure will have its own subplot for the different joints (e.g., left shoulder, right shoulder) to visualize the 
coherence and phase angle separately for each joint.

'''
WINDOW_SIZE_SECONDS = 10 
WINDOW_OVERLAP_SECONDS = 2
SELECTED_JOINT_NAMES = ['Head', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow']
JOINT_MOVEMENT_PATH = "/mnt/c/3HYPER FREEPLAY DV METRABs/MATLAB Keypoints 2/2D Keypoints Processed/3HYPER.025 FREEPLAY DV PROCESSED 2D Keypoints.mat"
SHOULDER_JOINT_PAIRS = list(product(SELECTED_JOINT_NAMES[1:3], repeat=2))
ELBOW_JOINT_PAIRS = list(product(SELECTED_JOINT_NAMES[3:], repeat=2))

def convert_seconds_to_frame(seconds, frame_rate):
    return int(seconds * frame_rate)

def make_sliding_windows(data, window_size_frames, overlap):
    step = window_size_frames - overlap
    num_windows = (data.shape[0] - window_size_frames) // step + 1
    windows = sliding_window_view(data, window_shape=window_size_frames)[::step]
    return windows[:num_windows]

def calculate_average_joint_movement(joint_data, selected_joint_names):
    selected_joint_data = np.array([joint_data[name] for name in selected_joint_names])
    average_joint_movement = np.mean(selected_joint_data, axis=0)
    return average_joint_movement

def plot_average_wtc(avg_wtc_windows: dict[str, np.ndarray], window_size_seconds: float, overlap_seconds: float, joint_name: str, joint_pairs: list[str], save_name: str):
    # add legend for head and shoulder joints
    num_windows = avg_wtc_windows[joint_name].shape[0]
    time_axis = np.arange(num_windows) * (window_size_seconds - overlap_seconds)
    coherence_axis = np.arange(0, 1, 0.1)
    phase_axis = np.arange(0, 360, 45)
    
    fig, ax = plt.subplots(4, 1, figsize=(12, 12))
    plt.suptitle(f'Windowed Average Wavelet Transform Coherence (WTC) of {joint_name} Keypoint in Dyad #25')
    
    for ax_idx, ax in enumerate(ax):
        ax.plot(time_axis, avg_wtc_windows[joint_pairs[ax_idx]], color='blue', alpha=0.9)
        ax.set_title(f'{joint_pairs[ax_idx]} WTC')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Coherence')
        ax.set_yticks(coherence_axis)
        ax.grid()
    
    plt.tight_layout()
    plt.show()
    # plt.savefig(save_name)
    
def plot_phase_binned(head_phase_binned, shoulder_phase_binned, phase_labels):
    x = np.arange(len(phase_labels))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, head_phase_binned, width, label='Head')
    plt.bar(x + width/2, shoulder_phase_binned, width, label='Shoulders')
    plt.xticks(x, phase_labels)
    plt.xlabel('Phase Relationship')
    plt.ylabel('Counts')
    plt.title('Phase Relationship Count for Head and Shoulders')
    plt.legend(loc='upper right')
    # plt.savefig("phase_binned_counts_2.png")

def main():
    # Load movement data 
    dyad_info = load_data_mat(JOINT_MOVEMENT_PATH)
    frame_rate = 27.49 # taken from database, but needs to be adjusted for every sample
    max_frames = max(len(dyad_info["Parent"]["Head"]), len(dyad_info["Infant"]["Head"]))
    dt = 1/frame_rate # period of the signal (s)
    s0 = 2 * dt  # smallest scale of the wavelet transform
    
    # Convert window size and overlap from seconds to frames
    window_size_frames = convert_seconds_to_frame(WINDOW_SIZE_SECONDS, frame_rate)
    overlap_frames = convert_seconds_to_frame(WINDOW_OVERLAP_SECONDS, frame_rate)
    
    # Calculate average joint movement for selected joints and add them back to dyad_info 
    # dyad_info["Infant"]["Shoulders Averaged"] = calculate_average_joint_movement(dyad_info["Infant"], SELECTED_JOINT_NAMES[1:]) 
    # dyad_info["Parent"]["Shoulders Averaged"] = calculate_average_joint_movement(dyad_info["Parent"], SELECTED_JOINT_NAMES[1:])
    
    # Compute wavelet coherence and phase angles for entire signal, average across FOI, then average across windows
    #  WTC and phase angles for the head and shoulders
    head_wtc_signal, head_phase_signal, head_coi, head_freqs, hsig = compute_wtc(dyad_info, SELECTED_JOINT_NAMES[0], s0, dt) 
    right_shoulder_wtc_signal, right_shoulder_phase_signal, right_shoulder_coi, right_shoulder_freqs, rsig = compute_wtc(dyad_info, "Right Shoulder", s0, dt) 
    left_shoulder_wtc_signal, left_shoulder_phase_signal, left_shoulder_coi, left_shoulder_freqs, lsig = compute_wtc(dyad_info, "Left Shoulder", s0, dt) 
    shoulder_wtc_signal = (right_shoulder_wtc_signal + left_shoulder_wtc_signal) / 2
    shoulder_phase_signal = circmean(np.stack([right_shoulder_phase_signal, left_shoulder_phase_signal], axis=0),axis=0)
    # shoulder_wtc_signal, shoulder_phase_signal, shoulder_coi, shoulder_freqs, ssig = compute_wtc(dyad_info, "Shoulders Averaged", s0, dt) 
    
    # Extract data from FOI 
    foi_indices = np.where((head_freqs >= 0.5) & (head_freqs <= 1.5))[0]
    head_wtc_foi = head_wtc_signal[foi_indices, :]
    head_phase_foi = head_phase_signal[foi_indices, :]
    shoulder_wtc_foi = shoulder_wtc_signal[foi_indices, :]
    shoulder_phase_foi = shoulder_phase_signal[foi_indices, :]
    
   # Average the WTC and phase angles across the frequency of interest (FOI) before windowing
    avg_head_wtc_foi = np.mean(head_wtc_foi, axis=0)
    avg_head_phase_foi = circmean(head_phase_foi, axis=0)
    avg_shoulder_wtc_foi = np.mean(shoulder_wtc_foi, axis=0)
    avg_shoulder_phase_foi = circmean(shoulder_phase_foi, axis=0)
    
    # Average the WTC and phase angles across the sliding windows
    avg_head_wtc_windowed = np.mean(make_sliding_windows(avg_head_wtc_foi, window_size_frames, overlap_frames), axis=1)
    avg_shoulder_wtc_windowed = np.mean(make_sliding_windows(avg_shoulder_wtc_foi, window_size_frames, overlap_frames), axis=1)
    avg_head_phase_windowed = circmean(make_sliding_windows(avg_head_phase_foi, window_size_frames, overlap_frames), low=-np.pi, high=np.pi, axis=1)
    avg_shoulder_phase_windowed = circmean(make_sliding_windows(avg_shoulder_phase_foi, window_size_frames, overlap_frames), low=-np.pi, high=np.pi, axis=1)
    
     # Convert phase angles from radians to degrees for easier interpretation 
    head_phase_dg, shoulder_phase_dg = np.degrees(avg_head_phase_windowed) % 360, np.degrees(avg_shoulder_phase_windowed) % 360
    
    avg_wtc_info = {
        "Head": avg_head_wtc_windowed,
        "Shoulder Average": avg_shoulder_wtc_windowed
    }
    
    avg_phase_info = {
        "Head": head_phase_dg,
        "Shoulder Average": shoulder_phase_dg
    }
    # Bin phase values to determine dominant phase relationship within each window
    phase_labels = ["In-Phase", "Parent-Lead", "Anti-Phase", "Infant-Lead"]
    head_phase_binned = []
    shoulder_phase_binned = []
    
    head_in_phase = np.sum((head_phase_dg <= 45) | (head_phase_dg > 315))
    head_parent_lead = np.sum((head_phase_dg > 45) & (head_phase_dg <= 135))
    head_anti_phase = np.sum((head_phase_dg > 135) & (head_phase_dg <= 225))
    head_infant_lead = np.sum((head_phase_dg > 225) & (head_phase_dg <= 315)) 
    head_phase_binned.extend([head_in_phase, head_parent_lead, head_anti_phase, head_infant_lead])
    
    shoulder_in_phase = np.sum((shoulder_phase_dg <= 45) | (shoulder_phase_dg > 315))
    shoulder_parent_lead = np.sum((shoulder_phase_dg > 45) & (shoulder_phase_dg < 135))
    shoulder_anti_phase = np.sum((shoulder_phase_dg >= 135) & (shoulder_phase_dg < 225))
    shoulder_infant_lead = np.sum((shoulder_phase_dg >= 225) & (shoulder_phase_dg < 315))
    shoulder_phase_binned.extend([shoulder_in_phase, shoulder_parent_lead, shoulder_anti_phase, shoulder_infant_lead])
    
    # head_phase_bin_info = dict(zip(phase_labels, head_phase_binned))
    # shoulder_phase_bin_info = dict(zip(phase_labels, shoulder_phase_binned))
    
    # Plot the windowed average WTC and phase angles

if __name__ == "__main__":
    main()
    



