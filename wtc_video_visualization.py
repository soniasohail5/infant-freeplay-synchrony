import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons, Slider, Button
from signal_postprocessing import lin_interp_threshold
from missing_gaps_stats import get_video_name, get_dyad_number
from single_dyad_wtc_analysis import load_data_mat, compute_wtc,DESIRED_JOINT_NAMES
from signal_visualization_multi import apply_medfilt_to_all_keypoints, VideoLoader

'''
Side-by-side visualization of WTC heatmap and skeletal overlays on raw video data to 
get a better understanding of how low and high coherence regions map to physical interactions 
between infant and parent subjects
'''
video_path = "/mnt/e/IN-PERSON EXPERIMENT RECORDINGS/3HYPER FREEPLAY/3HYPER DV FREEPLAY/3HYPER.025 FREEPLAY DV.mp4"
keypoints_path = "/mnt/c/3HYPER FREEPLAY DV METRABs/MATLAB Keypoints 2/2D Keypoints Processed/3HYPER.025 FREEPLAY DV PROCESSED 2D Keypoints.mat"

# General class for figures with multiple subplots that require synchronization (ie. need to load simultaneously)
class MultiDataSyncFigure:
    def __init__(self, num_rows:int, num_cols:int, dyad_number:int, video_path:str, keypoints_path:str, 
                 total_frames:int, x_label:list, y_label:list, titles:dict):
        # video params
        self.video = VideoLoader(video_path)
        self.video._open_video()
        self.fps = self.video.fps
        self.frames = total_frames
        
        # data params
        self.id = dyad_number
        self.dyad_info = load_data_mat(keypoints_path)
        
        # figure/plot params
        self.fig, self.ax = plt.subplots(num_rows, num_cols, figsize=(12, 10), sharex=True)
        self.suptitle = titles["Figure Title"]
        self.x_label = x_label
        self.y_label = y_label
        
        self.x_lim = max(len(self.dyad_info["Infant"]["Head"]), len(self.dyad_info["Parent"]["Head"]))
        self.y_lim = max(self.dyad_info["Infant"]["Head"].max(), self.dyad_info["Parent"]["Head"].max())
        self.timestamps = np.arange(self.frames)/self.fps
        
    def get_wtc(self, joint_name:str):
        dt = 1/self.fps 
        s0 = 2 * dt
        
        selected_joint = joint_name if joint_name is in DESIRED_JOINT_NAMES else "Head"
        wtc, a_wct, coi, freq, sig  = compute_wtc(self.dyad_info, selected_joint, s0, dt)
        self.wtc_data = {"WTC": wtc, "Phase Angles": a_wct, "COI": coi, 
                         "Frequency": freq, "Significance": sig}
        
    def place_video_bg(self):
        
    def initial_plot(self):
    def update_figure(self, playback_speed:int):
    def clear_figure(self):
    
        
        

        
        
        
        
    
        
        
        
    
        
        
    
        
        

        
        
        
        
        
        
        
        
        
        
        
        








