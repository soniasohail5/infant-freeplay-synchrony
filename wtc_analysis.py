import os 
import copy
import pandas as pd
import numpy as np
import pycwt as wavelet
from matplotlib import pyplot
from scipy.io import loadmat
from signal_postprocessing import replace_missing
from missing_gap_stats import get_video_name, get_dyad_number
from pycwt.helpers import find 

'''
Synchrony analysis of dyads using Wavelet Transform Coherence (WTC)
- Done for each joint of each dyad
- GCWT can be used at some point to combine related joints to 
minimize the number of computations done

5/26/26 Note: Only use data that has no NaN values (for demonstratiom purposes)
- remember to bring this concern up at the next weekly meeting
- one approach to solve this would be to break up the signal into its non-NaN segments, run CWT on each non-NaN segment
and concatencate the coherence and CWT values before plotting
'''

def load_data_mat(keypoints_path):
# Loads the keypoints from the processed .mat files
# Output is a dictionary with the infant and parent keypoints
def convert_data_to_df(collection_of_keypoints):
# Takes the dictionary of loaded dyad keypoints and converts them into a DataFrame 
# Output is a pandas DataFrame
def extract_dyad_df(dyad_number, data_df):
# Pulls the infant and parent keypoints for a specific dyad from the dataframe
def compute_wtc(dyad_number, dyad_df):
# Calculates the continuous wavelet transform (CWT) for the infant and parent joint timeseries
# Uses CWT from infant and parent to calculate coherence (formula can be found in the Fujiwara paper)
def compute_cross_wt(dyad_number, dyad_df):
def plot_wtc(dyad_number, dyad_df):
# One figure has 3 subplots:
# (1) Original time series signals of infant and parent joint 
# (2) Results from cross wavelet transform (raw covariance)
# (3) Results from coherence (normalization of cross wavelet transform)


    



