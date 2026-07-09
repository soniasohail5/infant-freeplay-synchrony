import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import CheckButtons, Slider, Button
from missing_gaps_stats import get_video_name, get_dyad_number, import_data
from single_dyad_wtc_analysis import load_data_mat, compute_wtc, DESIRED_JOINT_NAMES
from signal_visualization_multi import VideoLoader

'''
Side-by-side visualization of WTC heatmap and skeletal overlays on raw video data to 
get a better understanding of how low and high coherence regions map to physical interactions 
between infant and parent subjects
'''

DESIRED_JOINT_INDICES = [12, 13, 14, 15, 16, 17, 18, 19]
JOINT_INDEX_ASSOCIATION = dict(zip(DESIRED_JOINT_NAMES, DESIRED_JOINT_INDICES))
video_path = "/mnt/e/IN-PERSON EXPERIMENT RECORDINGS/3HYPER FREEPLAY/3HYPER DV FREEPLAY/3HYPER.025 FREEPLAY DV.mp4"
joint_movement_path = "/mnt/c/3HYPER FREEPLAY DV METRABs/MATLAB Keypoints 2/2D Keypoints Processed/3HYPER.025 FREEPLAY DV PROCESSED 2D Keypoints.mat"
keypoints_path = "/mnt/c/3HYPER FREEPLAY DV METRABs/MATLAB Keypoints 2/2D Keypoints/3HYPER.025 FREEPLAY DV EXTRACTED 2D Keypoints.mat"

# Class for figures with multiple subplots that require synchronization (ie. need to load simultaneously)
class MultiDataSyncFigure:
    def __init__(self, num_rows:int, num_cols:int, dyad_number:int, video_path:str, movement_path:str, keypoints_path:str, 
                 total_frames:int, x_label:list, y_label:list, titles:dict):
        # video params
        self.video = VideoLoader(video_path)
        self.video._open_video()
        self.fps = self.video.fps
        self.frames = total_frames
        self.playback_speed = self.video.fps
        self.is_playing = False
        
        # data params
        self.id = dyad_number
        self.dyad_info = load_data_mat(movement_path) # 1D signal 
        self.joint_info = import_data(keypoints_path) # 2D signal (joint, x/y, frame)
        
        # figure/plot params
        self.fig, self.ax = plt.subplots(num_rows, num_cols, figsize=(10, 10))
        self.suptitle = titles["Figure Title"]
        self.sb_titles = titles["Subplot Titles"]
        self.x_labels = x_label
        self.y_labels = y_label
        self.timer = self.fig.canvas.new_timer(interval=int(1000/self.fps))
        
        self.x_lim = self.video.width
        self.y_lim = self.video.height
        self.timestamps = np.arange(self.frames)/self.fps
        self.current_joint = DESIRED_JOINT_NAMES[0]
        self.infant_selected_joint = DESIRED_JOINT_NAMES[0]
        self.parent_selected_joint = DESIRED_JOINT_NAMES[0]
        
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        
    def _on_close(self, event):
        self.timer.stop()
        self.is_playing = False
        
    def advance_frame(self):
        current_frame = int(self.slider.val)
        next_frame = current_frame + max(2 * self.video.fps, int(self.playback_speed))
        
        if next_frame >= self.frames:
            self.is_playing = False
            self.timer.stop()
            return
        
        self.slider.set_val(next_frame)
    
    def update_time_text(self):
        current_frame = int(self.slider.val)
        current_time = current_frame/self.fps
        self.time_text.set_text(f'Time:{current_time:.2f}s | Speed: {self.playback_speed/self.video.fps:.4}x')
     
    def get_wtc(self):
        dt = 1/self.fps 
        s0 = 2 * dt
        self.all_wtc_data = {}
        
        # calculate wtc for all joints beforehand to avoid pauses in videoplay with another joint is selected
        for infant_joint in DESIRED_JOINT_NAMES:
            for parent_joint in DESIRED_JOINT_NAMES:
                joint_pair  = [infant_joint, parent_joint]
            
                if infant_joint == parent_joint:
                    joint_pair_str = parent_joint 
                else:
                    joint_pair_str = infant_joint + ", " + parent_joint
                    
                wtc, a_wct, coi, freq, sig  = compute_wtc(self.dyad_info, joint_pair, s0, dt)
                wtc_data = {"WTC": wtc, "Phase Angles": a_wct, "COI": coi, 
                            "Frequency": freq, "Significance": sig}
                self.all_wtc_data[joint_pair_str] = wtc_data
            
    def set_axes(self):
        # places the titles, axes, and buttons for each plot 
        self.fig.suptitle(self.suptitle)
        plt.subplots_adjust(top=0.93)
        
        if len(self.x_labels) != len(self.y_labels):
            print("Mismatch between number of x and y labels.")
            return
        
        if len(self.ax) != len(self.sb_titles):
            print("Mismatch between number of axes and subplot titles.")
            return
        
        # fig and axes 
        for (a, i, subtitle) in zip(self.ax, range(len(self.x_labels)), self.sb_titles):
            a.set_title(subtitle)
            a.set_xlabel(self.x_labels[i])
            a.set_ylabel(self.y_labels[i])
            a.set_xlim(0, self.x_lim)
            a.set_ylim(0, self.y_lim)
            
        # slider
        ax_slider = plt.axes([0.15, 0.04, 0.7, 0.02])
        slider = Slider(ax_slider, 'Frame', valmin=0, valmax=max(1, self.frames - 1),
                        valinit=0, valstep=1)
            
        # time display
        time_text = self.fig.text(0.5, 0.05, f'Time: 0.00s | Speed: {self.playback_speed/self.video.fps:.4}x',
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
            self.playback_speed = min(16.5, self.playback_speed + 0.25)
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
        infant_check_ax = plt.axes([0.75, 0.73, 0.15, 0.15])
        parent_check_ax = plt.axes([0.25, 0.45, 0.15, 0.15])
        self.infant_check = CheckButtons(infant_check_ax, DESIRED_JOINT_NAMES)
        self.parent_check = CheckButtons(parent_check_ax, DESIRED_JOINT_NAMES)
        
        def infant_check_callback(label):
            # ensures that only one joint is selected at a time
            for joint in DESIRED_JOINT_NAMES:
                if joint != label and self.infant_check.get_status()[DESIRED_JOINT_NAMES.index(joint)]:
                   self.infant_check.set_active(DESIRED_JOINT_NAMES.index(joint))
                   
            # update current joint
            self.infant_selected_joint = label 
            
            if hasattr(self, 'wtc_mesh'):
                self.colorbar.remove()
                self.wtc_mesh.remove()
                self.contour_regions.remove()
                self.coi_fill.remove()
                del self.contour_regions
                del self.colorbar
                del self.wtc_mesh
                del self.coi_fill
  
            # get current frame index from slider
            current_frame = int(slider.val)
            self.update_figure(current_frame)
            
        def parent_check_callback(label):
            # ensures that only one joint is selected at a time
            for joint in DESIRED_JOINT_NAMES:
                if joint != label and self.parent_check.get_status()[DESIRED_JOINT_NAMES.index(joint)]:
                   self.parent_check.set_active(DESIRED_JOINT_NAMES.index(joint))
                   
            # update current joint
            self.parent_selected_joint = label 
            
            if hasattr(self, 'wtc_mesh'):
                self.colorbar.remove()
                self.wtc_mesh.remove()
                self.contour_regions.remove()
                self.coi_fill.remove()
                del self.contour_regions
                del self.colorbar
                del self.wtc_mesh
                del self.coi_fill

            # get current frame index from slider
            current_frame = int(slider.val)
            self.update_figure(current_frame)
            
        self.infant_check.on_clicked(infant_check_callback)
        self.parent_check.on_clicked(parent_check_callback)

    def initialize_video(self, joint_name:str, frame_number:int = 0):
        # plots first frame with joints
        self.current_joint = joint_name
        self.get_wtc()  # needed before plot_wtc can access self.all_wtc_data
        self.draw_video_bg(ax_number=0, frame_number=frame_number)
        self.plot_joints(ax_number=0, frame_number=frame_number)
        self.plot_wtc(ax_number=1, frame_number=frame_number)
    
    def preload_frames(self, max_frames=500):
        # pre load frames into RAM before playback begins 
        self.frame_cache = {}
        for i in range(min(max_frames, self.frames)):
            self.frame_cache[i] = self.video.get_frame(i)
            
    def draw_video_bg(self, ax_number:int, frame_number:int):
        # plots frame overlay in the background
        
        if frame_number == 0 and not hasattr(self, 'wtc_axis_intitialized'):
            self.ax[ax_number].invert_yaxis()  # invert y-axis to match video coordinates
            
        if self.video:
            video_frame = self.video.get_frame(frame_number)
            video_frame = cv2.resize(video_frame, (640, 360)) # downsample before caching
            if video_frame is not None:
                if not hasattr(self, 'video_im'):
                    extent = [0, self.video.width, self.video.height, 0]
                    self.video_im = self.ax[ax_number].imshow(video_frame, extent=extent, aspect='auto', zorder=0)
                else:
                    self.video_im.set_data(video_frame)
                    
    def plot_joints(self, ax_number:int, frame_number:int):
        # plots joint keypoints
        joint_names = [self.infant_selected_joint, self.parent_selected_joint]
        joint_index = [JOINT_INDEX_ASSOCIATION[joint] for joint in joint_names]
        infant_joint_x, infant_joint_y = self.joint_info["infant"][joint_index[0], 0, frame_number], self.joint_info["infant"][joint_index[0], 1, frame_number]
        parent_joint_x, parent_joint_y = self.joint_info["parent"][joint_index[1], 0, frame_number], self.joint_info["parent"][joint_index[1], 1, frame_number]
        
        if not hasattr(self, 'infant_scatter'):
            self.infant_scatter = self.ax[ax_number].scatter(infant_joint_x, infant_joint_y, color='red', alpha=0.7)
            self.parent_scatter = self.ax[ax_number].scatter(parent_joint_x, parent_joint_y, color='blue', alpha=0.7)
        else:
            self.infant_scatter.set_offsets([[infant_joint_x, infant_joint_y]])
            self.parent_scatter.set_offsets([[parent_joint_x, parent_joint_y]])
        
    def plot_wtc(self, ax_number:int, frame_number:int):
        # plots wtc
        if self.infant_selected_joint == self.parent_selected_joint:
            joint_pair = self.infant_selected_joint
        else:
            joint_pair = self.infant_selected_joint + ", " + self.parent_selected_joint
            
        wtc_data = self.all_wtc_data[joint_pair]
        wtc = wtc_data["WTC"]
        freq = wtc_data["Frequency"]
        sig = wtc_data["Significance"]
        coi = wtc_data["COI"]
        phase_angles = wtc_data["Phase Angles"]
        period = 1/freq
        
        n_freq, n_time = wtc.shape
        wtc_timestamps = np.arange(n_time)/self.fps
        
        # rather than adding each new column to the wtc, 
        # mask future coherence values and unmask them as the video plays 
        wtc_masked = np.full_like(wtc, np.nan)
        wtc_masked[:, :frame_number] = wtc[:, :frame_number]
        
        sig_clean = np.nan_to_num(sig, nan=0.0)
        wtc_sig_plot = sig_clean[:, np.newaxis] * np.ones_like(wtc_masked)
        
        coi_masked = np.full_like(coi, np.nan, dtype=float)
        coi_masked[:frame_number] = 1/coi[:frame_number]
        
        if not hasattr(self, 'wtc_axis_intitialized'):
            # set up axes properties only once
            self.ax[ax_number].set_yscale('log', base=2)
            self.ax[ax_number].set_yticks([0.03, 0.06, 0.12, 0.25, 0.5, 1, 2, 4, 8])
            self.ax[ax_number].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            self.ax[ax_number].set_ylim([freq.min(), freq.max()])
            self.ax[ax_number].set_xlim([0, wtc_timestamps[-1]])
            
        if not hasattr(self, 'wtc_mesh'):
            # initialize the pcolormesh object 
            self.wtc_mesh = self.ax[ax_number].pcolormesh(wtc_timestamps, freq, wtc_masked, 
                                                          cmap='jet', vmin=0, vmax=1)
            self.colorbar = self.fig.colorbar(self.wtc_mesh, ax=self.ax[ax_number], label='Coherence')
            self.contour_regions = self.ax[ax_number].contour(wtc_timestamps, freq, wtc/wtc_sig_plot, levels=[1.0], colors='black', linewidths=1.5)
        else:
            self.wtc_mesh.set_array(wtc_masked.ravel())
            
        valid = ~np.isnan(coi_masked)
        if valid.any() and not hasattr(self, 'coi_filled'):
            self.coi_fill =  self.ax[ax_number].fill_between(wtc_timestamps[valid], coi_masked[valid], freq.min(), alpha=0.2, color='gray', hatch='x')
            
        if hasattr(self, 'phase_arrows'):
            self.phase_arrows.remove()
            del self.phase_arrows
            
        sig_mask = wtc > sig_clean[:, np.newaxis]
        t_skip = 50
        p_skip = 2
        
        if frame_number > 1:
            if frame_number % t_skip == 0 or not hasattr(self, 'phase_arrows'):
                t_indices = np.arange(0, min(frame_number, n_time), t_skip)
                p_indices = np.arange(0, len(freq), p_skip)
                t_grid, p_grid = np.meshgrid(wtc_timestamps[t_indices], freq[p_indices])
                
                u = np.cos(phase_angles[np.ix_(p_indices, t_indices)])
                v = np.sin(phase_angles[np.ix_(p_indices, t_indices)])
                mask = sig_mask[np.ix_(p_indices, t_indices)]
                
                if mask.any():
                    self.phase_arrows = self.ax[ax_number].quiver(t_grid[mask], p_grid[mask], u[mask], v[mask],
                                                                  units='width', pivot='mid', headwidth=3, width=0.002, scale=50, color='black', zorder=6)

    def update_figure(self, val:int):
        # for video quality 
        frame_number = int(val)
        self.draw_video_bg(ax_number=0, frame_number=frame_number)
        self.plot_joints(ax_number=0, frame_number=frame_number)
        self.plot_wtc(ax_number=1, frame_number=frame_number)
        self.update_time_text()
        self.ax[0].draw_artist(self.video_im)
        self.ax[0].draw_artist(self.infant_scatter) # redraw only video
        self.ax[0].draw_artist(self.parent_scatter) # redraw only scatter
        self.fig.canvas.blit(self.ax[0].bbox) 
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
    def clear_figure(self):
        # free up memory after video is finished playing or is interrupted
        self.timer.stop()
        self.video.video.release() if hasattr(self.video, 'video') else None
        plt.close(self.fig)

def main():
    titles = {
        "Figure Title": "WTC Visualization - Dyad #25",
        "Subplot Titles": ["Video", "Wavelet Coherence"]
    }
    
    fig_obj = MultiDataSyncFigure(
        num_rows=2, num_cols=1,
        dyad_number=25,
        video_path=video_path, 
        movement_path=joint_movement_path,
        keypoints_path=keypoints_path,
        total_frames=6607,
        x_label=["Time (s)", "Time (s)"],
        y_label=["", "Frequency (Hz)"],
        titles=titles
    )
    
    fig_obj.set_axes()
    fig_obj.initialize_video(joint_name=DESIRED_JOINT_NAMES[0])
    plt.show()

if __name__ == "__main__":
    main()
        
        

        
        
        
        
    
        
        
        
    
        
        
    
        
        

        
        
        
        
        
        
        
        
        
        
        
        








