import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons, Slider, Button
from signal_postprocessing import lin_interp_threshold
from missing_gaps_stats import get_video_name, get_dyad_number, import_data
from single_dyad_wtc_analysis import load_data_mat, compute_wtc, DESIRED_JOINT_NAMES
from signal_visualization_multi import apply_medfilt_to_all_keypoints, VideoLoader

'''
Side-by-side visualization of WTC heatmap and skeletal overlays on raw video data to 
get a better understanding of how low and high coherence regions map to physical interactions 
between infant and parent subjects
'''
DESIRED_JOINT_INDICES = [12, 13, 14, 15, 16, 17, 18, 19]
JOINT_INDEX_ASSOCIATION = dict(zip(DESIRED_JOINT_NAMES, DESIRED_JOINT_INDICES))
video_path = "/mnt/e/IN-PERSON EXPERIMENT RECORDINGS/3HYPER FREEPLAY/3HYPER DV FREEPLAY/3HYPER.025 FREEPLAY DV.mp4"
keypoints_path = "/mnt/c/3HYPER FREEPLAY DV METRABs/MATLAB Keypoints 2/2D Keypoints Processed/3HYPER.025 FREEPLAY DV PROCESSED 2D Keypoints.mat"

# Class for figures with multiple subplots that require synchronization (ie. need to load simultaneously)
class MultiDataSyncFigure:
    def __init__(self, num_rows:int, num_cols:int, dyad_number:int, video_path:str, keypoints_path:str, 
                 total_frames:int, x_label:list, y_label:list, titles:dict):
        # video params
        self.video = VideoLoader(video_path)
        self.video._open_video()
        self.fps = self.video.fps
        self.frames = total_frames
        self.playback_speed = 1.0
        self.is_playing = False
        
        # data params
        self.id = dyad_number
        self.dyad_info = load_data_mat(keypoints_path) # 1D signal 
        self.joint_info = import_data(keypoints_path) # 2D joint coordinates
        
        # figure/plot params
        self.fig, self.ax = plt.subplots(num_rows, num_cols, figsize=(12, 10), sharex=True)
        self.suptitle = titles["Figure Title"]
        self.sb_titles = titles["Subplot Titles"]
        self.x_labels = x_label
        self.y_labels = y_label
        self.timer = self.fig.canvas.new_timer(interval=int(1000/self.fps))
        
        self.x_lim = self.video.width
        self.y_lim = self.video.height
        self.timestamps = np.arange(self.frames)/self.fps
        self.current_joint = DESIRED_JOINT_NAMES[0]
        
        
    def advance_frame(self):
        current_frame = int(self.slider.val)
        next_frame = current_frame + max(1, int(self.playback_speed))
        
        if next_frame >= self.frames:
            self.btn_pause_callback()
            self.timer_stop()
            return
        
        self.slider.set_val(next_frame)
    
    def update_time_text(self):
        current_frame = int(self.slider.val)
        current_time = current_frame/self.fps
        self.time_text.set_text(f'Time:{current_time:.2f}s | Speed: {self.playback_speed}x')
    
    def preprocess_joint_data(self):
        
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
        
        if len(self.x_labels) != len(self.y_labels):
            print("Mismatch between number of x and y labels.")
            return
        
        if len(self.ax) != len(self.sb_titles):
            print("Mismatch between number of axes and subplot titles.")
            return
        
        # fig and axes 
        for (a, i, subtitle) in zip(self.ax, range(len(self.x_labels)), self.sb_titles):
            a.set_title(self.sb_titles[subtitle])
            a.set_xlabel(self.x_labels[i])
            a.set_ylabel(self.y_labels[i])
            a.set_xlim(self.x_lim)
            a.set_ylim(self.y_lim)
            
        # slider
        ax_slider = plt.axes([0.15, 0.08, 0.7, 0.02])
        slider = Slider(ax_slider, 'Frame', valmin=0, valmax=max(1, self.frames - 1),
                        valinit=0, valstep=1)
            
        # time display
        time_text = self.fig.text(0.5, 0.03, f'Time: 0.00s | Speed: {playback_speed}x',
                             ha='center', fontsize=12)
        self.time_text = time_text
        
        # buttons 
        ax_play       = plt.axes([0.15, 0.01, 0.08, 0.03])
        ax_pause      = plt.axes([0.24, 0.01, 0.08, 0.03])
        ax_reset      = plt.axes([0.33, 0.01, 0.08, 0.03])
        ax_step_back  = plt.axes([0.50, 0.01, 0.08, 0.03])
        ax_step_fwd   = plt.axes([0.59, 0.01, 0.08, 0.03])
        ax_speed_down = plt.axes([0.68, 0.01, 0.08, 0.03])
        ax_speed_up   = plt.axes([0.77, 0.01, 0.08, 0.03])

        self.btn_play       = Button(ax_play,'Play')
        self.btn_pause      = Button(ax_pause,'Pause')
        self.btn_reset      = Button(ax_reset,'Reset')
        self.btn_step_back  = Button(ax_step_back, '< Step')
        self.btn_step_fwd   = Button(ax_step_fwd, 'Step >')
        self.btn_speed_down = Button(ax_speed_down, 'Speed -')
        self.btn_speed_up   = Button(ax_speed_up, 'Speed +')
        
        def btn_play_callback(event):
            if not self.is_playing: 
                self.is_playing = True 
                self.timer.start()
            
        def btn_pause_callback(event):
            self.is_playing = False 
            self.timer.stop()
            
        def btn_reset_callback(event):
            self.is_playing = False
            self.timer.stop()
            self.slider.set_val(0)
            
        def step_back_callback(event):
            self.is_playing = False
            self.timer.stop()
            current_frame = int(self.slider.val)
            self.slider.set_val(max(0, current_frame - 1))
            
        def step_fwd_callback(event):
            self.is_playing = False
            self.timer.stop()
            current_frame = int(self.slider.val)
            self.slider.set_val(min(self.frames - 1, current_frame + 1))
            
        def btn_speed_down_callback(event):
            self.playback_speed = max(0.25, self.playback_speed - 0.25)
            self.update_time_text()
            
        def btn_speed_up_callback(event):
            self.playback_speed = max(4.0, self.playback_speed + 0.25)
            self.update_time_text()
            
        self.btn_play.on_clicked(btn_play_callback)
        self.btn_pause.on_clicked(btn_pause_callback)
        self.btn_reset.on_clicked(btn_reset_callback)
        self.btn_step_back.on_clicked(step_back_callback)
        self.btn_step_fwd.on_clicked(step_fwd_callback)
        self.btn_speed_down.on_clicked(btn_speed_down_callback)
        self.btn_speed_up.on_clicked(btn_speed_up_callback)
        self.timer.add_callback(self.advance_frame)

        self.slider = slider  
        slider.on_changed(self.update_figure)
            
        # checkboxes for joints
        check_ax = plt.axes([0.05, 0.4, 0.15, 0.15])
        self.check = CheckButtons(check_ax, DESIRED_JOINT_NAMES)
        
        def check_callback(label):
            # ensures that only one joint is selected at a time
            for joint in DESIRED_JOINT_NAMES:
                if joint != label and self.check.get_status()[DESIRED_JOINT_NAMES.index(joint)]:
                   self.check.set_active(DESIRED_JOINT_NAMES.index(joint))
            
            # update current joint
            self.current_joint = label 
            
            if hasattr(self, 'wtc_mesh'):
                del self.wtc_mesh
  
            # get current frame index from slider
            current_frame = int(slider.val)
            self.update_figure(current_frame)
            
        self.check.on_clicked(check_callback)
        

    def initialize_video(self, joint_name:str, frame_number:int = 0):
        # plots first frame with joints
        infant_joint_data, parent_joint_data = self.joint_info["infant"], self.joint_info["parent"]
        infant_movement_data, parent_movement_data = self.dyad_info["Infant"], self.joint_info["Parent"]
        self.current_joint = joint_name
        self.get_wtc()  # needed before plot_wtc can access self.all_wtc_data
        self.draw_video_bg(ax_number=0, frame_number=frame_number)
        self.plot_joints(ax_number=0, frame_number=frame_number)
        self.plot_wtc(ax_number=1, frame_number=frame_number)
        
    def draw_video_bg(self, ax_number:int, frame_number:int):
        # plots frame overlay in the background
        if self.video:
            video_frame = self.video.get_frame(frame_number)
            if video_frame is not None:
                extent = [0, self.video.width, self.video.height, 0]
                self.ax[ax_number].imshow(video_frame, extent=extent, aspect='auto', zorder=0)
                
    def plot_joints(self, ax_number:int, frame_number:int):
        # plots joint keypoints
        joint_name = self.current_joint
        joint_index = JOINT_INDEX_ASSOCIATION[joint_name]
        infant_joint_x, infant_joint_y = self.joint_info["infant"][joint_index, 0, frame_number], self.joint_info["infant"][joint_index, 1, frame_number]
        parent_joint_x, parent_joint_y = self.joint_info["parent"][joint_index, 0, frame_number], self.joint_info["parent"][joint_index, 1, frame_number]
        
        self.ax[ax_number].scatter(infant_joint_x, infant_joint_y, color='red', alpha=0.7)
        self.ax[ax_number].scatter(parent_joint_x, parent_joint_y, color='blue', alpha=0.7)
        
    def plot_wtc(self, ax_number:int, frame_number:int):
        # plots wtc
        joint_name = self.current_joint
        wtc_data = self.all_wtc_data[joint_name]
        wtc = wtc_data["WTC"]
        freq = wtc_data["Frequency"]
        period = 1/freq
        
        n_freq, n_time = wtc.shape
        
        # rather than adding each new column to the wtc, 
        # mask future coherence values and unmask them as the video plays 
        wtc_masked = np.full_like(wtc, np.nan)
        wtc_masked[:, :frame_number] = wtc[:, :frame_number]
        
        if not hasattr(self, 'wtc_mesh'):
            # initialize the pcolormesh object 
            self.wtc_mesh = self.ax[ax_number].pcolormesh(self.timestamps, period, wtc_masked, 
                                                          cmap='jet', vmin=0, vmax=1, shading='auto')
            self.ax[ax_number].set_yscale('log', base=2)
            self.ax[ax_number].invert_yaxis()
            self.fig.colorbar(self.wtc_mesh, ax=self.ax[ax_number], label='Coherence')
        else:
            self.wtc_mesh.set_array(wtc_masked[:-1, :-1].ravel())
            
    def update_figure(self, val:int):
        # for video quality 
        frame_number = int(val)
        self.draw_video_bg(ax_number=0, frame_number=frame_number)
        self.plot_joints(ax_number=0, frame_number=frame_number)
        self.plot_wtc(ax_number=1, frame_number=frame_number)
        self.update_time_text()
        self.fig.canvas.draw_idle()
        
    def clear_figure(self):
        # free up memory after video is finished playing or is interrupted
        self.timer.stop()
        self.video.video.release() if hasattr(self.video, 'video') else None
        plt.close(self.fig)
    
        
        

        
        
        
        
    
        
        
        
    
        
        
    
        
        

        
        
        
        
        
        
        
        
        
        
        
        








