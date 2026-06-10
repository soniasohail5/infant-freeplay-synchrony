import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons, Slider, Button
from signal_postprocessing import lin_interp_threshold
from missing_gaps_stats import get_video_name, get_dyad_number
from single_dyad_wtc_analysis import load_data_mat, compute_wtc, DESIRED_JOINT_NAMES
from signal_visualization_multi import apply_medfilt_to_all_keypoints, VideoLoader

'''
Side-by-side visualization of WTC heatmap and skeletal overlays on raw video data to 
get a better understanding of how low and high coherence regions map to physical interactions 
between infant and parent subjects

'''
video_path = "/mnt/e/IN-PERSON EXPERIMENT RECORDINGS/3HYPER FREEPLAY/3HYPER DV FREEPLAY/3HYPER.025 FREEPLAY DV.mp4"
keypoints_path = "/mnt/c/3HYPER FREEPLAY DV METRABs/MATLAB Keypoints 2/2D Keypoints Processed/3HYPER.025 FREEPLAY DV PROCESSED 2D Keypoints.mat"

class MultiDataSyncFigure:
    def __init__(self, dyad_number, video_path, infant_keypoints, parent_keypoints, total_frames, fps=30):
        self.fig, self.ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        self.id = dyad_number
        self.video_path = video_path
        self.infant = np.array(infant_keypoints) if not isinstance(infant_keypoints, np.ndarray) else infant_keypoints
        self.parent = np.array(parent_keypoints) if not isinstance(parent_keypoints, np.ndarray) else parent_keypoints
        self.fps = fps
        self.frames = total_frames
        
        
        
        
        
        








