import os
import numpy as np
import random as rand
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.signal as signal 
from scipy.interpolate import CubicSpline
from itertools import groupby
from signal_postprocessing import replace_missing, butterworth_filter, movmad_filter, linear_interp, lin_interp_threshold

# Plotting the signals (head, elbows, shoulders) from 20 randomly selected dyads 
# Separate plots for x and y coordinates as well as for each filter (original vs filtered)

def find_missing_segments_indices(data):
    # input is assumed to be 1d (corresponding to each joint signal in one direction)
    indices = np.argwhere(np.isnan(data)).flatten()
    
    if len(indices) == 0:
        return []
    
    if len(indices) == 1:
        return [indices]

    segments = []
    
    start_index = indices[0]
    end_index = indices[1]
    
    current_index = start_index
    
    for i, index in enumerate(indices[1:]):
        next_index = index

        if (next_index - current_index) == 1:
            # next_index is part of the same segment 
            current_index = next_index
        else:
            # segment is found, where next_index is the first index of a new segment
            end_index = current_index
            segment = np.arange(start_index, end_index + 1)
            segments.append(segment)
            start_index = next_index
            current_index = next_index
    
    # save the last segment
    segments.append(np.arange(start_index, current_index + 1))
    return segments

def find_known_data_segments(data, max_frames, nan_segments):
    # input is assumed to be 1d (one joint signal in singular direction)
    max_index = max_frames - 1
    known_data_segments = []
    
    if nan_segments[0][0] > 0:
        first_segment  = np.arange(0, nan_segments[0][0])
        known_data_segments.append(first_segment)
        
    for i in range(len(nan_segments) - 1):
        start_index = nan_segments[i][-1] + 1
        end_index = nan_segments[i + 1][0] - 1
        known_segment = np.arange(start_index, end_index)
        known_data_segments.append(known_segment)
        
    if nan_segments[-1][-1] < max_index:
        last_segment = np.arange(nan_segments[-1][-1] + 1, max_index + 1)
        known_data_segments.append(last_segment)
        
    return known_data_segments
        
def segment_signal(data):
    # divide signal into segments with non-NaN values (for filtering)
    segments = []
    for is_nan, group in groupby(data, key=np.isnan):
        if not is_nan:
            segments.append(list(group))
            
    return segments
    
def segmented_butterworth(segmented_signal, order, cutoff, fs):
    filtered_signal = []
    for segment in segmented_signal:
        filt_segment = butterworth_filter(segment, order, cutoff, fs)
        filtered_signal.append(filt_segment)
        
    return filtered_signal

def determine_max_shape(list_of_signals): # edit this function since first two dimensions are the same for every dyad
    max_shape = [0, 0, 0]
    for arr in list_of_signals:
        for i in range(3):
            if arr.shape[i] > max_shape[i]:
                max_shape[i] = arr.shape[i]
                
    return max_shape

def padding_arrays(list_of_signals, max_shape):
    padded_arrays = []
    for arr in list_of_signals:
        pad_width = ((0, max_shape[0] - arr.shape[0]),  (0, max_shape[1] - arr.shape[1]), (0, max_shape[2] - arr.shape[2]))
        padded_arr = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
        padded_arrays.append(padded_arr)
        
    return padded_arrays

keypoints_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints"
global_infant_joint_movement = []
global_parent_joint_movement = []

# Random sampling of dyads for plotting and analysis
rand.seed(42)

dyad_ids = [20, 21, 22, 23, 24, 25, 27, 30, 32, 33, 34, 37, 38, 39, 40, 41, 42, 43, 45, 47, 
            48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 62, 63, 64, 67, 68, 69, 70, 71, 73, 74, 75, 
            77, 78, 79, 80, 87, 91, 92, 93, 95, 96, 97, 98, 99, 100, 102, 103, 104, 108,
            109, 110, 111, 112, 113, 114, 117, 118, 119, 120, 121]

random_dyads = rand.sample(dyad_ids, 2)

# Selecting sample dyads for filter testing 
selected_dyads = [108]
excluded_dyads = [57, 76, 78, 112]

def main():

    # Aggregating infant and parent joint data for plotting data distribution
    for id in selected_dyads:
        if id >= 100:
            keypoints = "3HYPER." + str(id) + " FREEPLAY DV EXTRACTED 2D Keypoints"
        else:
            keypoints = "3HYPER.0" + str(id) + " FREEPLAY DV EXTRACTED 2D Keypoints"
            
        keypoints_path = os.path.join(keypoints_dir, keypoints)
        keypoints = sio.loadmat(keypoints_path)
        
        print(f"Plotting {keypoints_path} .....")
        
        # Determine parent and infant signals from MATLAB file using head keypoints
        labels = []
        keypoints_all = []
        head_y_values = []
        
        for label, person in keypoints.items():
            if not label.startswith('person'):
                continue
            if label == 'person_2_2d':
                continue

            kp = np.array(person)
            labels.append(label)
            keypoints_all.append(kp)

            head_y = np.nan
            for f in range(kp.shape[1]):
                y = kp[15, 1, f]
                if not np.isnan(y):
                    head_z = abs(y)
                    break

            head_y_values.append(head_z)

        if len(head_y_values) < 2:
            print(" Skipping file (not enough people)")
            continue

        infant_index = int(np.nanargmax(head_y_values))
        parent_index = int(np.nanargmin(head_y_values))
        
        # Plotting signals of randomly selected dyads
        
        infant_signal = keypoints_all[infant_index]
        parent_signal = keypoints_all[parent_index]
        
        joint_names = ["Pelvis", "Left Hip", "Right Hip", "Spine 1", "Left Knee", "Right Knee", "Spine 2", "Left Ankle", "Right Ankle", "Spine 3", "Left Toe", 
                    "Right Toe", "Neck", "Left Calf", "Right Calf", "Head", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", 
                    "Left Hand", "Right Hand"]
        
        joint_indices = [15]
        
        coords = ['X', 'Y']
        # EDIT 3/17/2026: indices to plot only specific segments of the signal (for exploratory analysis)
        fs = 30
        cutoff = 3
        
        # Replace missing data with NaN
        infant_signal, infant_nan = replace_missing(infant_signal)
        parent_signal, parent_nan = replace_missing(parent_signal)
        
        infant_filtered_data = np.zeros_like(infant_signal)
        parent_filtered_data = np.zeros_like(parent_signal)
        
        raw_signals = [infant_signal, parent_signal]
        subject = ["Infant", "Parent"]
        
        if infant_nan or parent_nan:
            print("Raw signals contain NaN values.")
        else: 
            print("Signals have non-NaN values.")
        
        for signal, label in zip(raw_signals, subject):
            for coordinate in range(len(coords)):
                coordinate_name = coords[coordinate]
                for joint in range(len(joint_indices)):
                    joint_index = joint_indices[joint]
                    joint_name = joint_names[joint_index]
                    
                    fig1, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
                    # fig2, ax2 = plt.subplots(2, 1, figsize=(10, 10), sharex=False)
                    
                    for a in ax:
                        a.set_ylabel('Position')
                        ax[-1].set_xlabel('Time (s)')
                        
                    '''
                    for a in ax2:
                        a.set_ylabel('Magnitude')
                        ax2[-1].set_xlabel('Frequency')
                    '''
                
                    if label == "Infant":
                        fig1.suptitle(f'{joint_name} Signal for Infant #{id} (Original vs Interpolated), coordinate={coords[coordinate]}-direction', fontweight="bold")
                        # fig2.suptitle(f'FFT of {joint_name} Signal for Infant #{id} (Original vs Filtered), coordinate={coords[coordinate]}-direction')
                        nan_flag = infant_nan
                    else:
                        fig1.suptitle(f'{joint_name} Signal for Parent #{id} (Original vs Interpolated), coordinate={coords[coordinate]}-direction', fontweight="bold")
                        # fig2.suptitle(f'FFT of {joint_name} Signal for Parent #{id} (Original vs Filtered), coordinate={coords[coordinate]}-direction')
                        nan_flag = parent_nan
                        
                    original_signal = signal[joint_index, coordinate, :]
                    
                    max_frames = len(original_signal)
                    lininterp_signal = linear_interp(original_signal)
                    lininterpthreshold_signal = lin_interp_threshold(original_signal, 12)
                    # med_signal = movmad_filter(lininterp_signal, 30)
                    
                    demeaned_signal = lininterp_signal - np.mean(lininterp_signal)
                    original_rfft = np.fft.rfft(demeaned_signal)
                    original_rfreq = np.fft.rfftfreq(len(lininterp_signal), 1/fs)
                    
                    filtered_signal = butterworth_filter(lininterp_signal, 4, cutoff, 30)
                    timestamps = np.arange(max_frames)
                    
                    demeaned_filtered = filtered_signal - np.mean(filtered_signal)
                    filtered_rfft = np.fft.rfft(demeaned_filtered)
                    filtered_rfreq = np.fft.rfftfreq(len(filtered_signal), 1/fs)
                
                    ax[0].plot(timestamps, original_signal, color='gray')
                    ax[0].set_title('Original Signal')
                    
                    ax[1].plot(timestamps, movmad_filter(lininterpthreshold_signal, 25), color='red')
                    ax[1].set_title(f'Signal After Linear Interpolation w/ Threshold (f=12)')

                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    main()
                
                
                    
        
        

    
                    



        
        
        
        
        
        
        
        




