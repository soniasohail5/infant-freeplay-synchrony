import os 
import copy
import math
import pandas as pd
import numpy as np
import pycwt as wavelet
from matplotlib import pyplot
from scipy.io import loadmat
from pycwt.helpers import find 
from signal_postprocessing import replace_missing
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

def compute_wtc(dyad_info, joint_name, s0, dt, dj=1/12, significance_level=0.95):
# Calculates the continuous wavelet transform (CWT) for the infant and parent joint timeseries
# Uses CWT from infant and parent to calculate coherence (formula can be found in the Fujiwara paper)
    
    infant_signal = dyad_info["Infant"][joint_name]
    parent_signal = dyad_info["Parent"][joint_name]
    
    wct, a_wct, coi, freq, sig = wavelet.wct([infant_signal, parent_signal], dt, dj, s0, 
                                             wavelet='morlet', significance_level=significance_level)
    return wct, a_wct, coi, freq, sig
    
def compute_cross_wt(dyad_info, joint_name, s0, dt, dj, significance_level=0.95):
# Calculates the cross wavelet transform (XWT) between the infant and parent time series signals 
# based on their individual CWTs
# Establishes the raw covariance/power between the time series joint signals

    infant_signal = dyad_info["Infant"][joint_name]
    parent_signal = dyad_info["Parent"][joint_name]
    
    xwt, x, coi, freqs, sig = wavelet.xwt([infant_signal, parent_signal], dt, dj, s0, 
                                          wavelet='morlet', significance_level=significance_level)
    return xwt, x, coi, freqs, sig

# def plot_wtc(dyad_number, dyad_df, joint_name, dt, dj, s0):
# One figure has 3 subplots:
# (1) Original time series signals of infant and parent joint 
# (2) Results from cross wavelet transform (raw covariance)
# (3) Results from coherence (normalization of cross wavelet transform)

def main():
    
    dt = 1/30
    s0 = 2 * dt
    dyad_info = load_data_mat(keypoints_dir)
    # print(type(dyad_info["Infant"]["Head"]))
    
    xwt, x, coi, freqs, sig = compute_cross_wt(dyad_info, "Head", s0, dt, 1/12)
    wct, a_wct, coi, freq, sig = compute_wtc(dyad_info, "Head", s0, dt, 1/12)
    
if __name__ == "__main__":
    main()
    
    
    





