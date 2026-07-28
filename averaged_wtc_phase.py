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
FREQ_LOW = 0.5 # in Hz
FREQ_HIGH = 2.0 # in Hz
SELECTED_JOINT_NAMES = ['Head', 'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow']
JOINT_MOVEMENT_PATH = "/mnt/c/3HYPER FREEPLAY DV METRABs/MATLAB Keypoints 2/2D Keypoints Processed/3HYPER.025 FREEPLAY DV PROCESSED 2D Keypoints.mat"
SHOULDER_JOINT_PAIRS = list(product(SELECTED_JOINT_NAMES[1:3], repeat=2))
ELBOW_JOINT_PAIRS = list(product(SELECTED_JOINT_NAMES[3:], repeat=2))

def convert_seconds_to_frame(seconds: int, frame_rate: float):
    return int(seconds * frame_rate)

def make_sliding_windows(data: np.ndarray, window_size_frames: int, overlap: int):
    step = window_size_frames - overlap
    num_windows = (data.shape[0] - window_size_frames) // step + 1
    windows = sliding_window_view(data, window_shape=window_size_frames)[::step]
    return windows[:num_windows]

def calculate_average_joint_movement(joint_data: dict[np.ndarray], selected_joint_names: list):
    selected_joint_data = np.array([joint_data[name] for name in selected_joint_names])
    average_joint_movement = np.mean(selected_joint_data, axis=0)
    return average_joint_movement

def plot_average_metric(avg_qt_windows: dict[str, np.ndarray], window_size_seconds: float, overlap_seconds: float, joint_name: str, joint_pairs: list[str], save_name: str, qt_label: str
                        ,y_ticks: list[float]):
    
    labels = [f"{joint_pair[0]}-{joint_pair[1]}" for joint_pair in joint_pairs]
    num_windows = avg_qt_windows[labels[0]].shape[0]
    time_axis = np.arange(num_windows) * (window_size_seconds - overlap_seconds)
    
    fig, ax = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(f'Windowed Average Wavelet Transform Coherence (WTC) of {joint_name} in Dyad #25')
    
    for ax_idx, ax in enumerate(ax):
        ax.plot(time_axis, avg_qt_windows[labels[ax_idx]], color='blue', alpha=0.9)
        ax.set_title(f'{qt_label} of {labels[ax_idx]}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(qt_label)
        ax.set_yticks(y_ticks)
        ax.grid()
    
    plt.tight_layout()
    plt.savefig(save_name)
    
def plot_phase_binned(head_phase_binned: list[np.int64], shoulder_phase_binned: dict[str, list[np.int64]], elbow_phase_binned: dict[str, list[np.int64]], phase_labels: list[np.str_]):
    x = np.arange(len(phase_labels))
    width = 0.35
    
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle('Phase Relationship Count for Head Movement')
    plt.bar(x, head_phase_binned, width, label='Head')
    plt.xlabel('Phase Relationship')
    plt.ylabel('Counts')
    plt.xticks(x, phase_labels)
    
    fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, shoulder_pair in zip(axes.flatten(), shoulder_phase_binned):
        ax.bar(x, shoulder_phase_binned[shoulder_pair], width, label=shoulder_pair)
        ax.set_title(f'{shoulder_pair}')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels)
        ax.set_ylabel('Counts')
        
    fig1.suptitle('Phase Relationship Count for Shoulder Joint Movement')
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    for ax, elbow_pair in zip(axes2.flatten(), elbow_phase_binned):
        ax.bar(x, elbow_phase_binned[elbow_pair], width, label=elbow_pair)
        ax.set_title(f'{elbow_pair}')
        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels)
        ax.set_ylabel('Counts')
        
    fig2.suptitle('Phase Relationship Count for Elbow Joint Movement')

    fig.savefig("phase_binned_counts_head.png")
    fig1.savefig("phase_binned_counts_shoulders.png")
    fig2.savefig("phase_binned_counts_elbows.png")
    
    plt.close()

def main():
    # Load movement data 
    dyad_info = load_data_mat(JOINT_MOVEMENT_PATH)
    frame_rate = 27.49 # taken from database, but needs to be adjusted for every sample
    max_frames = max(len(dyad_info["Parent"]["Head"]), len(dyad_info["Infant"]["Head"]))
    dt = 1/frame_rate # period of the signal (s)
    s0 = 2 * dt  # smallest scale of the wavelet transform
    
    print(SHOULDER_JOINT_PAIRS)
    print(ELBOW_JOINT_PAIRS)
    
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
    lr_shoulder_wtc_signal, lr_shoulder_phase_signal, lr_shoulder_coi, lr_shoulder_freqs, lrsig = compute_wtc(dyad_info, list(SHOULDER_JOINT_PAIRS[1]), s0, dt) 
    rl_shoulder_wtc_signal, rl_shoulder_phase_signal, rl_shoulder_coi, rl_shoulder_freqs, rlsig = compute_wtc(dyad_info, list(SHOULDER_JOINT_PAIRS[2]), s0, dt) 
    
    left_elbow_wtc_signal, left_elbow_phase_signal, left_elbow_coi, left_elbow_freqs, lesig = compute_wtc(dyad_info, list(ELBOW_JOINT_PAIRS[0]), s0, dt) 
    right_elbow_wtc_signal , right_elbow_phase_signal, right_elbow_coi, right_elbow_freqs, resig = compute_wtc(dyad_info, list(ELBOW_JOINT_PAIRS[3]), s0, dt) 
    lr_elbow_wtc_signal, lr_elbow_phase_signal, lr_elbow_coi, lr_elbow_freqs, lresig = compute_wtc(dyad_info, list(ELBOW_JOINT_PAIRS[1]), s0, dt) 
    rl_elbow_wtc_signal, rl_elbow_phase_signal, rl_elbow_coi, rl_elbow_freqs, rlesig = compute_wtc(dyad_info, list(ELBOW_JOINT_PAIRS[2]), s0, dt) 
    
    all_shoulder_wtc_signals = [left_shoulder_wtc_signal, right_shoulder_wtc_signal, lr_shoulder_wtc_signal, rl_shoulder_wtc_signal]
    all_shoulder_phase_signals = [left_shoulder_phase_signal, right_shoulder_phase_signal, lr_shoulder_phase_signal, rl_shoulder_phase_signal]
    all_elbow_wtc_signals = [left_elbow_wtc_signal, right_elbow_wtc_signal, lr_elbow_wtc_signal, rl_elbow_wtc_signal]
    all_elbow_phase_signals = [left_elbow_phase_signal, right_elbow_phase_signal, lr_elbow_phase_signal, rl_elbow_phase_signal]
    
    # shoulder_wtc_signal = (right_shoulder_wtc_signal + left_shoulder_wtc_signal) / 2
    # shoulder_phase_signal = circmean(np.stack([right_shoulder_phase_signal, left_shoulder_phase_signal], axis=0),axis=0)
    # shoulder_wtc_signal, shoulder_phase_signal, shoulder_coi, shoulder_freqs, ssig = compute_wtc(dyad_info, "Shoulders Averaged", s0, dt) 
    
    # Extract data from FOI 
    foi_indices = np.where((head_freqs >= FREQ_LOW) & (head_freqs <= FREQ_HIGH))[0]
    head_wtc_foi = head_wtc_signal[foi_indices, :]
    head_phase_foi = head_phase_signal[foi_indices, :]
    
    all_shoulder_wtc_foi = [wtc[foi_indices, :] for wtc in all_shoulder_wtc_signals]
    all_shoulder_phase_foi = [phase[foi_indices, :] for phase in all_shoulder_phase_signals]
    all_elbow_wtc_foi = [wtc[foi_indices, :] for wtc in all_elbow_wtc_signals]
    all_elbow_phase_foi = [phase[foi_indices, :] for phase in all_elbow_phase_signals]
        
    # shoulder_wtc_foi = shoulder_wtc_signal[foi_indices, :]
    # shoulder_phase_foi = shoulder_phase_signal[foi_indices, :]    
    
   # Average the WTC and phase angles across the frequency of interest (FOI) before windowing
    avg_head_wtc_foi = np.mean(head_wtc_foi, axis=0)
    avg_head_phase_foi = circmean(head_phase_foi, axis=0)
    # avg_shoulder_wtc_foi = np.mean(shoulder_wtc_foi, axis=0)
    # avg_shoulder_phase_foi = circmean(shoulder_phase_foi, axis=0)
    
    all_avg_shoulder_wtc_foi = [np.mean(wtc_foi, axis=0) for wtc_foi in all_shoulder_wtc_foi]
    all_avg_shoulder_phase_foi = [circmean(phase_foi, axis=0) for phase_foi in all_shoulder_phase_foi] 
    all_avg_elbow_wtc_foi = [np.mean(wtc_foi, axis=0) for wtc_foi in all_elbow_wtc_foi]
    all_avg_elbow_phase_foi = [circmean(phase_foi, axis=0) for phase_foi in all_elbow_phase_foi]
    
    # Average the WTC and phase angles across the sliding windows
    avg_head_wtc_windowed = np.mean(make_sliding_windows(avg_head_wtc_foi, window_size_frames, overlap_frames), axis=1)
    # avg_shoulder_wtc_windowed = np.mean(make_sliding_windows(avg_shoulder_wtc_foi, window_size_frames, overlap_frames), axis=1)
    avg_head_phase_windowed = circmean(make_sliding_windows(avg_head_phase_foi, window_size_frames, overlap_frames), low=-np.pi, high=np.pi, axis=1)
    # avg_shoulder_phase_windowed = circmean(make_sliding_windows(avg_shoulder_phase_foi, window_size_frames, overlap_frames), low=-np.pi, high=np.pi, axis=1)
    all_avg_shoulder_wtc_windowed = [np.mean(make_sliding_windows(avg_shoulder_wtc_foi, window_size_frames, overlap_frames), axis=1) for avg_shoulder_wtc_foi in all_avg_shoulder_wtc_foi]
    all_avg_shoulder_phase_windowed = [circmean(make_sliding_windows(avg_shoulder_phase_foi, window_size_frames, overlap_frames), low=-np.pi, high=np.pi, axis=1) for avg_shoulder_phase_foi in all_avg_shoulder_phase_foi]
    all_avg_elbow_wtc_windowed = [np.mean(make_sliding_windows(avg_elbow_wtc_foi, window_size_frames, overlap_frames), axis=1) for avg_elbow_wtc_foi in all_avg_elbow_wtc_foi]
    all_avg_elbow_phase_windowed = [circmean(make_sliding_windows(avg_elbow_phase_foi, window_size_frames, overlap_frames), low=-np.pi, high=np.pi, axis=1) for avg_elbow_phase_foi in all_avg_elbow_phase_foi]
    
     # Convert phase angles from radians to degrees for easier interpretation 
    head_phase_dg = np.degrees(avg_head_phase_windowed)
    # shoulder_phase_dg = np.degrees(avg_shoulder_phase_windowed) % 360
    all_shoulder_phase_dg = [(np.degrees(shoulder_phase)) for shoulder_phase in all_avg_shoulder_phase_windowed]
    all_elbow_phase_dg = [(np.degrees(elbow_phase)) for elbow_phase in all_avg_elbow_phase_windowed]
    
    avg_wtc_info = {
        "Head": avg_head_wtc_windowed,
    }
    
    avg_phase_info = {
        "Head": head_phase_dg,
    }
    
    for joint_pair, shoulder_wtc, shoulder_phase in zip(SHOULDER_JOINT_PAIRS, all_avg_shoulder_wtc_windowed, all_shoulder_phase_dg):
        label = f"{joint_pair[0]}-{joint_pair[1]}"
        avg_wtc_info[label] = shoulder_wtc
        avg_phase_info[label] = shoulder_phase
        
    for joint_pair, elbow_wtc, elbow_phase in zip(ELBOW_JOINT_PAIRS, all_avg_elbow_wtc_windowed, all_elbow_phase_dg):
        label = f"{joint_pair[0]}-{joint_pair[1]}"
        avg_wtc_info[label] = elbow_wtc
        avg_phase_info[label] = elbow_phase
    
    # Bin phase values to determine dominant phase relationship within each window
    phase_labels = ["In-Phase", "Parent-Lead", "Anti-Phase", "Infant-Lead"]
    head_phase_binned = []
    shoulder_phase_binned = {}
    elbow_phase_binned = {}
    
    head_in_phase = np.sum(( (head_phase_dg % 360 <= 45) | (head_phase_dg % 360 > 315)) )
    head_parent_lead = np.sum((head_phase_dg % 360 > 45) & (head_phase_dg % 360 <= 135))
    head_anti_phase = np.sum((head_phase_dg % 360 > 135) & (head_phase_dg % 360 <= 225))
    head_infant_lead = np.sum((head_phase_dg % 360 > 225) & (head_phase_dg % 360 <= 315)) 
    head_phase_binned.extend([head_in_phase, head_parent_lead, head_anti_phase, head_infant_lead])
    
    all_shoulder_in_phase = [np.sum((shoulder_phase_dg % 360 <= 45) | (shoulder_phase_dg % 360 > 315)) for shoulder_phase_dg in all_shoulder_phase_dg]
    all_shoulder_parent_lead =[np.sum((shoulder_phase_dg % 360 > 45) & (shoulder_phase_dg % 360 <= 135)) for shoulder_phase_dg in all_shoulder_phase_dg]
    all_shoulder_anti_phase = [np.sum((shoulder_phase_dg % 360 > 135) & (shoulder_phase_dg % 360 <= 225)) for shoulder_phase_dg in all_shoulder_phase_dg]
    all_shoulder_infant_lead = [np.sum((shoulder_phase_dg % 360 > 225) & (shoulder_phase_dg % 360 <= 315)) for shoulder_phase_dg in all_shoulder_phase_dg]
    
    all_elbow_in_phase = [np.sum((elbow_phase_dg % 360 <= 45) | (elbow_phase_dg % 360 > 315)) for elbow_phase_dg in all_elbow_phase_dg]
    all_elbow_parent_lead =[np.sum((elbow_phase_dg % 360 > 45) & (elbow_phase_dg % 360 <= 135)) for elbow_phase_dg in all_elbow_phase_dg]
    all_elbow_anti_phase = [np.sum((elbow_phase_dg % 360 > 135) & (elbow_phase_dg % 360 <= 225)) for elbow_phase_dg in all_elbow_phase_dg]
    all_elbow_infant_lead = [np.sum((elbow_phase_dg % 360 > 225) & (elbow_phase_dg % 360 <= 315)) for elbow_phase_dg in all_elbow_phase_dg]

    for (joint_pair, in_phase, parent_lead, anti_phase, infant_lead) in zip(SHOULDER_JOINT_PAIRS, all_shoulder_in_phase, all_shoulder_parent_lead, all_shoulder_anti_phase, all_shoulder_infant_lead):
            label = f"{joint_pair[0]}-{joint_pair[1]}"
            shoulder_phase_binned[label] = [in_phase, parent_lead, anti_phase, infant_lead]

    for (joint_pair, in_phase, parent_lead, anti_phase, infant_lead) in zip(ELBOW_JOINT_PAIRS, all_elbow_in_phase, all_elbow_parent_lead, all_elbow_anti_phase, all_elbow_infant_lead):
            label = f"{joint_pair[0]}-{joint_pair[1]}"
            elbow_phase_binned[label] = [in_phase, parent_lead, anti_phase, infant_lead]
            
    print(type(shoulder_phase_binned))
        
    # head_phase_bin_info = dict(zip(phase_labels, head_phase_binned))
    # shoulder_phase_bin_info = dict(zip(phase_labels, shoulder_phase_binned))

    # Plot the windowed average WTC and phase angles
    plot_average_metric(avg_wtc_info, WINDOW_SIZE_SECONDS, WINDOW_OVERLAP_SECONDS, "Shoulders", SHOULDER_JOINT_PAIRS, "average_wtc_shoulders.png", "Coherence", y_ticks=np.arange(0, 1.1, 0.1))
    plot_average_metric(avg_wtc_info, WINDOW_SIZE_SECONDS, WINDOW_OVERLAP_SECONDS, "Elbows", ELBOW_JOINT_PAIRS, "average_wtc_elbows.png", "Coherence", y_ticks=np.arange(0, 1.1, 0.1))
    plot_average_metric(avg_phase_info, WINDOW_SIZE_SECONDS, WINDOW_OVERLAP_SECONDS, "Shoulders", SHOULDER_JOINT_PAIRS, "average_phase_shoulders.png", "Phase (degrees)", y_ticks=[-180, 0, 180])
    plot_average_metric(avg_phase_info, WINDOW_SIZE_SECONDS, WINDOW_OVERLAP_SECONDS, "Elbows", ELBOW_JOINT_PAIRS, "average_phase_elbows.png", "Phase (degrees)", y_ticks=[-180, 0, 180])
    
    plot_phase_binned(head_phase_binned, shoulder_phase_binned, elbow_phase_binned, phase_labels)

if __name__ == "__main__":
    main()
    



