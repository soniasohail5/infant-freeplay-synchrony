import os
import cv2
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
        self.sb_titles = titles["Subplot Titles"]
        self.x_labels = x_label
        self.y_labels = y_label
        
        self.x_lim = max(len(self.dyad_info["Infant"]["Head"]), len(self.dyad_info["Parent"]["Head"]))
        self.y_lim = max(self.dyad_info["Infant"]["Head"].max(), self.dyad_info["Parent"]["Head"].max())
        self.timestamps = np.arange(self.frames)/self.fps
    
    def get_wtc(self):
        dt = 1/self.fps 
        s0 = 2 * dt
        self.all_wtc_data = {}
        
        # calculate wtc for all joints beforehand to avoid pauses in videoplay with another joint is selected
        for joint_name in DESIRED_JOINT_NAMES:
            wtc, a_wct, coi, freq, sig  = compute_wtc(self.dyad_info, joint_name, s0, dt)
            wtc_data = {"WTC": wtc, "Phase Angles": a_wct, "COI": coi, 
                         "Frequency": freq, "Significance": sig}
            self.all_wtc_data[joint_name] = wtc_data
            
    def set_axes(self, playback_speed=1.0):
        # places the titles, axes, and buttons for each plot 
        self.fig.suptitle(self.suptitle)
        
        if len(self.x_label) != len(self.y_label):
            print("Mismatch between number of x and y labels.")
            return
        
        if len(self.ax) != len(self.sb_titles):
            print("Mismatch between number of axes and subplot titles.")
            return
        
        # fig and axes 
        for (a, i, subtitle) in zip(self.ax, range(len(self.x_labels)), self.sb_titles):
            self.ax[a].set_title(self.sb_titles[subtitle])
            self.ax[a].set_xlabel(self.x_labels[i])
            self.ax[a].set_ylabel(self.y_labels[i])
            self.ax[a].set_xlim(self.xlim)
            self.ax[a].set_ylim(self.ylim)
            
        # slider
        ax_slider = plt.axes([0.15, 0.08, 0.7, 0.02])
        slider = Slider(ax_slider, 'Frame', valmin=0, valmax=max(1, self.n_frames - 1),
                        valinit=0, valstep=1)
            
        # time display
        time_text = self.fig.text(0.5, 0.03, f'Time: 0.00s | Speed: {playback_speed}x',
                             ha='center', fontsize=12)
        # buttons 
        ax_play       = plt.axes([0.15, 0.01, 0.08, 0.03])
        ax_pause      = plt.axes([0.24, 0.01, 0.08, 0.03])
        ax_reset      = plt.axes([0.33, 0.01, 0.08, 0.03])
        ax_step_back  = plt.axes([0.50, 0.01, 0.08, 0.03])
        ax_step_fwd   = plt.axes([0.59, 0.01, 0.08, 0.03])
        ax_speed_down = plt.axes([0.68, 0.01, 0.08, 0.03])
        ax_speed_up   = plt.axes([0.77, 0.01, 0.08, 0.03])

        btn_play       = Button(ax_play,'Play')
        btn_pause      = Button(ax_pause,'Pause')
        btn_reset      = Button(ax_reset,'Reset')
        btn_step_back  = Button(ax_step_back, '< Step')
        btn_step_fwd   = Button(ax_step_fwd, 'Step >')
        btn_speed_down = Button(ax_speed_down, 'Speed -')
        btn_speed_up   = Button(ax_speed_up, 'Speed +')
        
        current_speed = playback_speed
        
    def initialize_video(self, joint_name:str, frame_number=0:int):
        # plots first frame with joints
    def draw_video_bg(self, frame_number:int):
        # plots frame overlay in the background
    def plot_joint(self, joint_name:str):
        # plots joint keypoints
    def update_figure(self, val:int, playback_speed=1.0:int,):
        # for video quality 
    def clear_figure(self):
        # free up memory after video is finished playing or is interrupted
        
    
        
        

        
        
        
        
    
        
        
        
    
        
        
    
        
        

        
        
        
        
        
        
        
        
        
        
        
        








