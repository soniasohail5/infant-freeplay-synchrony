import os
import numpy as np
import pycwt as wavelet
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import detrend
from missing_gaps_stats import get_video_name, get_dyad_number

'''
Synchrony analysis of dyads using Wavelet Transform Coherence (WTC)
- Done for each joint of each dyad
- GCWT can be used at some point to combine related joints to 
minimize the number of computations done

5/26/26 Note: Only use data that has no NaN values (for demonstration purposes)
- remember to bring this concern up at the next weekly meeting
- one approach to solve this would be to break up the signal into its non-NaN segments, run CWT on each non-NaN segment
and concatencate the coherence and CWT values before plotting
'''
mother = wavelet.Morlet(8)
keypoints_dir = "/mnt/c/3HYPER FREEPLAY DV METRABs/MATLAB Keypoints 2/2D Keypoints Processed/3HYPER.025 FREEPLAY DV PROCESSED 2D Keypoints.mat"
DESIRED_JOINT_NAMES = ["Neck", "Head", "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow"]

def load_data_mat(keypoints_path):
# Loads the keypoints from the processed .mat files
# Output is a dictionary with the infant and parent keypoints with the 
# associated dyad number

    dyad_info = {}
    dyad_name = get_video_name(keypoints_path)
    dyad_number = get_dyad_number(dyad_name)
    dyad_info_mat = loadmat(keypoints_path)
    
    infant_keypoints_timeseries = np.array(dyad_info_mat["Infant"])
    parent_keypoints_timeseries = np.array(dyad_info_mat["Parent"])
    
    infant_info = {}
    parent_info = {}
    
    dyad_info["Dyad Number"] = dyad_number
    
    for (joint, index) in zip(DESIRED_JOINT_NAMES, range(6)):
        infant_info[joint] = infant_keypoints_timeseries[index, :]
        parent_info[joint] = parent_keypoints_timeseries[index, :]
        
    dyad_info["Infant"] = infant_info
    dyad_info["Parent"] = parent_info
    
    return dyad_info
    
def preprocess_signal(signal):
# Detrend and normalize signal using Z-transform
# before running wavelet analysis functions on dyad
    
    signal = np.diff(signal)
    signal = detrend(signal, type='linear')
    signal = signal - np.mean(signal)
    signal = signal/np.std(signal, ddof=1)
    
    return signal
        
def compute_wtc(dyad_info, joint_name, s0, dt, dj=1/12, significance_level=0.95):
# Calculates the continuous wavelet transform (CWT) for the infant and parent joint timeseries
# Uses CWT from infant and parent to calculate coherence (formula can be found in the Fujiwara paper)
    
    infant_signal = preprocess_signal(dyad_info["Infant"][joint_name])
    parent_signal = preprocess_signal(dyad_info["Parent"][joint_name])
    
    wct, a_wct, coi, freq, sig = wavelet.wct(infant_signal, parent_signal, dt, dj, s0, 
                                             wavelet='morlet', significance_level=significance_level)
    return wct, a_wct, coi, freq, sig
    
def compute_cross_wt(dyad_info, joint_name, s0, dt, dj=1/12, significance_level=0.95):
# Calculates the cross wavelet transform (XWT) between the infant and parent time series signals 
# based on their individual CWTs
# Establishes the raw covariance/power between the time series joint signals

    infant_signal = preprocess_signal(dyad_info["Infant"][joint_name])
    parent_signal = preprocess_signal(dyad_info["Parent"][joint_name])
    
    xwt, coi, freqs, sig = wavelet.xwt(infant_signal, parent_signal, dt, dj, s0, 
                                          wavelet='morlet', significance_level=significance_level)
    return xwt, coi, freqs, sig

def plot_wtc_xwt(dyad_number, dyad_info, joint_name, dt, s0):
# One figure has 3 subplots: 
# (1) Original time series signals of infant and parent joint 
# (2) Results from cross wavelet transform (raw covariance)
# (3) Results from coherence (normalization of cross wavelet transform)

    # Get original signals 
    infant_signal = dyad_info["Infant"][joint_name]
    parent_signal = dyad_info["Parent"][joint_name]

    # Compute WCT and XWT before setting up figure w/ plots
    wct, a_wct, wct_coi, freq, sig = compute_wtc(dyad_info, joint_name, s0, dt)
    xwt, xwt_coi, xfreqs, sig_xwt = compute_cross_wt(dyad_info, joint_name, s0, dt)
    
    # Initialize time/frequency scales
    max_frames = max(len(infant_signal), len(parent_signal))
    timestamps = np.arange(max_frames - 1) * dt
    wct_period = 1/freq
    xwt_period = 1/xfreqs
    
    # Set up figure and respective subplots
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(f"Wavelet Analysis of Dyad #{dyad_number}")
    
    # Subplot 1: Original position signals in image space
    ax[0].set_title("Original Signal")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Position (px)")
    ax[0].plot(timestamps, infant_signal, color='blue', label='Infant')
    ax[0].plot(timestamps, parent_signal, color='red', label='Parent')
    ax[0].legend(loc='upper right')
    
    # Subplot 2: Cross Wavelet Transform
    ax[1].set_title("Cross Wavelet Transform")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Period (s)")
    ax[1].set_ylim([xwt_period.min(), xwt_period.max()])
    ax[1].invert_yaxis()
    ax[1].set_yscale('log', base=2)
    
    im = ax[1].pcolormesh(timestamps, xwt_period, np.abs(xwt), 
                    cmap='jet', norm=matplotlib.colors.LogNorm())
    ax[1].fill_between(timestamps, xwt_coi, xwt_period.max(), alpha=0.3, 
                       color='gray', hatch='x', label='COI')
    ax[1].contour(timestamps, xwt_period, sig_xwt, levels=[1.0], colors='black', linewidths=1.5)
    fig.colorbar(im, ax=ax[1], label='Cross Wavelet Power')
    
    # Subplot 2: Wavelet Coherence
    
def main():

    dyad_info = load_data_mat(keypoints_dir)
    dyad_number = dyad_info["Dyad Number"]
    
    infant_signal = dyad_info["Infant"]["Head"]
    parent_signal = dyad_info["Parent"]["Head"]
    
    max_frames = max(len(infant_signal), len(parent_signal))
    dt = max_frames/240 # assuming all videos are exactly 4 minutes long (not always true, but can be resolved in batch analyses later)
    s0 = 2 * dt
    
    plot_wtc_xwt(dyad_number, dyad_info, "Head", dt, s0)

if __name__ == "__main__":
    main()
    
    
    





