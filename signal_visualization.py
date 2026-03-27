import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
import matplotlib.patches as mpatches
import cv2
import scipy.io as sio
from scipy import signal
import scipy.interpolate as interp
from signal_postprocessing import linear_interp

VIDEO_DURATION = 240 # in seconds

plt.rcParams['animation.ffmpeg_path'] = '/mnt/c/ffmpeg/bin/ffmpeg.exe'

# Joint edges for skeleton connectivity
JOINT_EDGES = [
    (1, 4), (1, 0), (2, 5), (2, 0), (3, 6), (3, 0),
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
    (9, 13), (9, 14),
    (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23),
]

JOINT_GROUP_COLORS = {
    1: '#e41a1c',  # red - head, spine1, neck
    2: '#377eb8',  # blue - hips, spines
    3: '#4daf4a',  # green - shoulders, elbows
    4: '#ff7f00',  # orange - wrists, hands
}

JOINT_GROUPS = [
    [3, 12, 15],
    [1, 2, 6, 9],
    [16, 17, 18, 19],
    [20, 21, 22, 23]
]

def get_joint_color(joint_idx):
    for group_id, indices in enumerate(JOINT_GROUPS, start=1):
        if joint_idx in indices:
            return JOINT_GROUP_COLORS[group_id]
    return '#999999'  # gray for ungrouped joints

# Cubic spline interpolation
def spline_interp(sig):
    sig = sig.copy()
    t = np.arange(len(sig))
    good = ~np.isnan(sig)

    if good.sum() < 4:
        return sig

    f = interp.interp1d(
        t[good],
        sig[good],
        kind='cubic',
        bounds_error=False,
        fill_value="extrapolate"
    )

    sig[~good] = f(t[~good])
    return sig

# ============== VIDEO LOADER ==============

class VideoLoader:
    """Helper class to load and cache video frames"""
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = None
        self.frames_cache = {}
        self.total_frames = 0
        self.fps = 30
        self.width = 0
        self.height = 0
        
        if video_path:
            self._open_video()
    
    def _open_video(self):
        """Open video and get metadata"""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Loaded video: {self.total_frames} frames, {self.fps} fps, {self.width}x{self.height}")
    
    def get_frame(self, frame_idx):
        """Get a specific frame (with caching)"""
        if frame_idx in self.frames_cache:
            return self.frames_cache[frame_idx]
        
        if self.cap is None:
            return None
        
        # Seek to frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames_cache[frame_idx] = frame_rgb
            return frame_rgb
        return None
    
    def __del__(self):
        """Clean up video capture"""
        if self.cap is not None:
            self.cap.release()


# ============== SKELETAL VISUALIZER ==============

class SkeletalVisualizer:
    """
    Interactive skeletal visualization comparing raw vs filtered data.
    Supports both animation playback and frame-by-frame scrubbing.
    Can overlay skeleton on original video frames.
    """
    
    def __init__(self, raw_data, filtered_data, fs=30, title="Skeletal Comparison", video_path=None):
        """
        Args:
            raw_data: Original keypoints, shape (n_joints, 2, n_frames)
            filtered_data: Filtered keypoints, same shape
            fs: Sampling frequency (fps)
            title: Figure title
            video_path: Optional path to source video file
        """
        self.raw = np.array(raw_data, dtype=float)
        self.filtered = np.array(filtered_data, dtype=float)
        self.fs = fs
        self.title = title
        
        self.n_joints = self.raw.shape[0]
        self.n_frames = self.raw.shape[2]
        self.current_frame = 0
        self.playing = False
        self.anim = None
        
        # Load video if provided
        self.video_loader = VideoLoader(video_path) if video_path else None
        
        # Compute data bounds for consistent axis limits
        all_data = np.concatenate([self.raw, self.filtered], axis=2)
        valid_x = all_data[:, 0, :][~np.isnan(all_data[:, 0, :])]
        valid_y = all_data[:, 1, :][~np.isnan(all_data[:, 1, :])]
        
        # If video is loaded, use video dimensions
        if self.video_loader:
            self.xlim = (0, self.video_loader.width)
            self.ylim = (0, self.video_loader.height)
        else:
            margin = 50
            self.xlim = (np.min(valid_x) - margin, np.max(valid_x) + margin)
            self.ylim = (np.min(valid_y) - margin, np.max(valid_y) + margin)
    
    def _draw_skeleton(self, ax, data, frame, color='blue', alpha=1.0, label=None, show_video_bg=False):
        """Draw a single skeleton frame, optionally on video background"""
        
        # Draw video frame as background if available
        if show_video_bg and self.video_loader:
            video_frame = self.video_loader.get_frame(frame)
            if video_frame is not None:
                ax.imshow(video_frame, extent=[0, self.video_loader.width, 
                                               self.video_loader.height, 0],
                         aspect='auto', zorder=0)
        
        x = data[:, 0, frame]
        y = data[:, 1, frame]
        
        # Draw edges with group-based coloring
        for i_start, i_end in JOINT_EDGES:
            if i_start < len(x) and i_end < len(x):
                if not (np.isnan(x[i_start]) or np.isnan(x[i_end])):
                    # Determine edge color based on the joints it connects
                    if color == 'multi':
                        # Use the color of the start joint (or end joint if start is ungrouped)
                        edge_color = get_joint_color(i_start)
                        if edge_color == '#999999':  # If start joint is ungrouped, try end joint
                            edge_color = get_joint_color(i_end)
                    else:
                        edge_color = color
                    
                edge_color=color    
                ax.plot([x[i_start], x[i_end]], [y[i_start], y[i_end]], color=edge_color, linewidth=2, alpha=alpha)
        
        # Draw joints with group colors
        for j in range(len(x)):
            if not np.isnan(x[j]):
                jcolor = get_joint_color(j) if color == 'multi' else color
                ax.scatter(x[j], y[j], c=jcolor, s=50, alpha=alpha, zorder=5)
        
        if label:
            ax.scatter([], [], c=color, label=label)
    
    def show_frame(self, frame_idx, show_video=False):
        """Display a single frame comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{self.title} - Frame {frame_idx} ({frame_idx/self.fs:.2f}s)')
        
        # Raw skeleton
        axes[0].set_title('Original (Raw)')
        self._draw_skeleton(axes[0], self.raw, frame_idx, color='green', show_video_bg=show_video)
        
        # Filtered skeleton
        axes[1].set_title('Filtered')
        self._draw_skeleton(axes[1], self.filtered, frame_idx, color='blue', show_video_bg=show_video)
        self._draw_skeleton(axes[1], self.raw, frame_idx, color='gray', show_video_bg=show_video )
        
        # Overlay comparison
        if show_video and self.video_loader:
            video_frame = self.video_loader.get_frame(frame_idx)
            if video_frame is not None:
                axes[2].imshow(video_frame, extent=[0, self.video_loader.width,
                                                    self.video_loader.height, 0],
                              aspect='auto', zorder=0)
        self._draw_skeleton(axes[0], self.raw, frame_idx, color='green', alpha=0.5, label='Raw')
        self._draw_skeleton(axes[1], self.raw, frame_idx, color='gray', alpha=0.6)
        self._draw_skeleton(axes[1], self.filtered, frame_idx, color='blue', alpha=0.8, label='Filtered')
        axes[2].legend()
        
        for ax in axes:
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
        
        plt.tight_layout()
        plt.show()
    
    def show_trajectory(self, joint_idx, start_frame=0, end_frame=None, step=1, show_video=False):
        """Show the trajectory of a specific joint over time"""
        if end_frame is None:
            end_frame = self.n_frames
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{self.title} - Joint {joint_idx} Trajectory '
                     f'(Frames {start_frame}-{end_frame})')
        
        frames = range(start_frame, end_frame, step)
        cmap = plt.cm.viridis
        colors = [cmap(i / len(frames)) for i in range(len(frames))]
        
        for ax, data, title in [(axes[0], self.raw, 'Original'),
                                 (axes[1], self.filtered, 'Filtered')]:
            ax.set_title(title)
            
            # Plot trajectory points colored by time
            for i, frame in enumerate(frames):
                x = data[joint_idx, 0, frame]
                y = data[joint_idx, 1, frame]
                if not np.isnan(x):
                    ax.scatter(x, y, c=[colors[i]], s=20, alpha=0.7)
            
            # Connect with lines
            x_traj = data[joint_idx, 0, start_frame:end_frame:step]
            y_traj = data[joint_idx, 1, start_frame:end_frame:step]
            valid = ~(np.isnan(x_traj) | np.isnan(y_traj))
            ax.plot(x_traj[valid], y_traj[valid], 'k-', alpha=0.3, linewidth=1)
            
            if show_video:
                # Add skeleton in background (will be updated in animation)
                ax.skeleton_artists = []
            
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
        
        # Colorbar for time
        sm = plt.cm.ScalarMappable(cmap=cmap, 
                                    norm=plt.Normalize(start_frame/self.fs, end_frame/self.fs))
        cbar = plt.colorbar(sm, ax=axes, shrink=0.8)
        cbar.set_label('Time (s)')
        
        # Add animation if requested
        if show_video:
            current_frame = [start_frame]  # Mutable container for closure
            
            def update_frame(frame_offset):
                frame = start_frame + frame_offset
                if frame >= end_frame:
                    return
                current_frame[0] = frame
                
                # Clear old skeleton artists
                for ax in axes:
                    for artist in ax.skeleton_artists:
                        artist.remove()
                    ax.skeleton_artists = []
                
                # Draw skeletons at current frame
                self._draw_skeleton_artists(axes[0], self.raw, frame, color='gray', alpha=0.3)
                self._draw_skeleton_artists(axes[1], self.filtered, frame, color='blue', alpha=0.3)
                
                # Update title with current frame
                fig.suptitle(f'{self.title} - Joint {joint_idx} Trajectory - Frame {frame} ({frame/self.fs:.2f}s)')
                
                return axes[0].skeleton_artists + axes[1].skeleton_artists
            
            # Create animation
            anim = FuncAnimation(fig, update_frame, frames=end_frame-start_frame, 
                               interval=1000/self.fs, blit=False, repeat=True)
        
        plt.tight_layout()
        plt.show()
    
    def _draw_skeleton_artists(self, ax, data, frame, color='blue', alpha=1.0):
        """Draw skeleton and store artists for animation updates"""
        x = data[:, 0, frame]
        y = data[:, 1, frame]
        
        if not hasattr(ax, 'skeleton_artists'):
            ax.skeleton_artists = []
        
        # Draw edges
        for i_start, i_end in JOINT_EDGES:
            if i_start < len(x) and i_end < len(x):
                if not (np.isnan(x[i_start]) or np.isnan(x[i_end])):
                    if color == 'multi':
                        edge_color = get_joint_color(i_start)
                        if edge_color == '#999999':
                            edge_color = get_joint_color(i_end)
                    else:
                        edge_color = color
                    
                    line, = ax.plot([x[i_start], x[i_end]], 
                                   [y[i_start], y[i_end]],
                                   color=edge_color, linewidth=2, alpha=alpha)
                    ax.skeleton_artists.append(line)
        
        # Draw joints
        for j in range(len(x)):
            if not np.isnan(x[j]):
                jcolor = get_joint_color(j) if color == 'multi' else color
                scatter = ax.scatter(x[j], y[j], c=jcolor, s=50, alpha=alpha, zorder=5)
                ax.skeleton_artists.append(scatter)
    
    def interactive_player(self, show_video=False, playback_speed=30.0):
        """
        Launch interactive player with playback controls and scrubbing
        
        Args:
            show_video: Whether to show video background
            playback_speed: Playback speed multiplier (1.0=normal, 2.0=2x speed, 0.5=half speed)
        """
        
        # Validate data first
        if self.n_frames < 2:
            print(f"Error: Need at least 2 frames, got {self.n_frames}")
            return
        
        if np.all(np.isnan(self.raw)) and np.all(np.isnan(self.filtered)):
            print("Error: All data is NaN")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        plt.subplots_adjust(bottom=0.25)
        
        title_suffix = " (with video)" if show_video and self.video_loader else ""
        speed_suffix = f" [{playback_speed}x speed]" if playback_speed != 1.0 else ""
        fig.suptitle(self.title + title_suffix + speed_suffix)
        axes[0].set_title('Original')
        axes[1].set_title('Filtered')
        
        for ax in axes:
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
        
        # Initial draw
        self._update_display(axes, 0, show_video=show_video)
        
        # Frame slider
        ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
        slider = Slider(
            ax_slider, 'Frame', 
            valmin=0, 
            valmax=max(1, self.n_frames - 1),
            valinit=0, 
            valstep=1
        )
            
        # Time display
        time_text = fig.text(0.5, 0.05, f'Time: 0.00s', 
                            ha='center', fontsize=12)
            
        # Buttons
        ax_play = plt.axes([0.15, 0.02, 0.08, 0.04])
        ax_pause = plt.axes([0.24, 0.02, 0.08, 0.04])
        ax_reset = plt.axes([0.33, 0.02, 0.08, 0.04])
        ax_step_back = plt.axes([0.50, 0.02, 0.08, 0.04])
        ax_step_fwd = plt.axes([0.59, 0.02, 0.08, 0.04])
        ax_speed_down = plt.axes([0.68, 0.02, 0.08, 0.04])
        ax_speed_up = plt.axes([0.77, 0.02, 0.08, 0.04])
        
        btn_play = Button(ax_play, 'Play')
        btn_pause = Button(ax_pause, 'Pause')
        btn_reset = Button(ax_reset, 'Reset')
        btn_step_back = Button(ax_step_back, '< Step')
        btn_step_fwd = Button(ax_step_fwd, 'Step >')
        btn_speed_down = Button(ax_speed_down, 'Speed -')
        btn_speed_up = Button(ax_speed_up, 'Speed +')
        
        # Current playback speed (mutable for closure)
        current_speed = [playback_speed]
        
        def update_slider(val):
            frame = int(slider.val)
            self._update_display(axes, frame, show_video=show_video)
            time_text.set_text(f'Time: {frame/self.fs:.2f}s | Speed: {current_speed[0]}x')
            fig.canvas.draw_idle()
            
        def play(event):
            self.playing = True
            self._animate(fig, axes, slider, time_text, show_video, current_speed)
        
        def pause(event):
            self.playing = False
        
        def reset(event):
            self.playing = False
            slider.set_val(0)
        
        def step_back(event):
            new_val = max(0, slider.val - 1)
            slider.set_val(new_val)
        
        def step_fwd(event):
            new_val = min(self.n_frames - 1, slider.val + 1)
            slider.set_val(new_val)
        
        def speed_down(event):
            current_speed[0] = max(0.25, current_speed[0] - 0.25)
            time_text.set_text(f'Time: {int(slider.val)/self.fs:.2f}s | Speed: {current_speed[0]}x')
            fig.suptitle(self.title + title_suffix + f" [{current_speed[0]}x speed]")
            fig.canvas.draw_idle()
            # Restart animation if playing
            if self.playing:
                self._animate(fig, axes, slider, time_text, show_video, current_speed)
        
        def speed_up(event):
            current_speed[0] = min(4.0, current_speed[0] + 0.25)
            time_text.set_text(f'Time: {int(slider.val)/self.fs:.2f}s | Speed: {current_speed[0]}x')
            fig.suptitle(self.title + title_suffix + f" [{current_speed[0]}x speed]")
            fig.canvas.draw_idle()
            # Restart animation if playing
            if self.playing:
                self._animate(fig, axes, slider, time_text, show_video, current_speed)
        
        slider.on_changed(update_slider)
        btn_play.on_clicked(play)
        btn_pause.on_clicked(pause)
        btn_reset.on_clicked(reset)
        btn_step_back.on_clicked(step_back)
        btn_step_fwd.on_clicked(step_fwd)
        btn_speed_down.on_clicked(speed_down)
        btn_speed_up.on_clicked(speed_up)
        
        # Legend for joint groups
        legend_patches = [mpatches.Patch(color=c, label=f'Group {g}') 
                            for g, c in JOINT_GROUP_COLORS.items()]
        fig.legend(handles=legend_patches, loc='upper right', fontsize=9)
        
        plt.show()

    def _update_display(self, axes, frame, show_video=False):
        """Update all three axes for a given frame"""
        for ax in axes:
            ax.clear()
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
        
        axes[0].set_title('Original')
        axes[1].set_title('Filtered')
        
        self._draw_skeleton(axes[0], self.raw, frame, color='green', show_video_bg=show_video)
        
        # Overlay with video background
        if show_video and self.video_loader:
            video_frame = self.video_loader.get_frame(frame)
            if video_frame is not None:
                axes[1].imshow(video_frame, extent=[0, self.video_loader.width,
                                                    self.video_loader.height, 0],
                              aspect='auto', zorder=0)
        self._draw_skeleton(axes[1], self.filtered, frame, color='blue', alpha=0.5)
        self._draw_skeleton(axes[1], self.raw, frame, color='gray', alpha=0.5)

    def _animate(self, fig, axes, slider, time_text, show_video=False, current_speed=None):
        """Animation loop with adjustable speed"""
        if current_speed is None:
            current_speed = [4.0]
            
        def update(frame):
            if not self.playing:
                return
            current = int(slider.val)
            next_frame = (current + 1) % self.n_frames
            slider.set_val(next_frame)
        
        # Adjust interval based on playback speed
        interval = (1000 / self.fs) / current_speed[0]  # ms per frame, adjusted by speed
        
        # Stop old animation if exists
        if self.anim is not None:
            self.anim.event_source.stop()
        
        self.anim = FuncAnimation(fig, update, interval=interval, 
                                    cache_frame_data=False)
        fig.canvas.draw()

    def save_animation(self, output_path, start_frame=0, end_frame=None, 
                        fps=None, dpi=100):
        """Save animation to video file"""
        if end_frame is None:
            end_frame = self.n_frames
        if fps is None:
            fps = self.fs
            
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(self.title)
        
        for ax in axes:
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
        
        def init():
            return []
        
        def animate(frame_idx):
            frame = start_frame + frame_idx
            self._update_display(axes, frame)
            fig.suptitle(f'{self.title} - Frame {frame} ({frame/self.fs:.2f}s)')
            return []
            
        n_anim_frames = end_frame - start_frame
        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=n_anim_frames, interval=1000/fps, blit=False)
        
        print(f"Saving animation to {output_path}...")
        anim.save(output_path, writer='ffmpeg', fps=fps, dpi=dpi)
        print("Done!")
        plt.close(fig)


def visualize_filtering_result(raw_data, filtered_data, fs=30, 
                                mode='interactive', video_path=None, 
                                playback_speed=30.0, **kwargs):
    
    viz = SkeletalVisualizer(raw_data, filtered_data, fs, video_path=video_path)
    
    if mode == 'interactive':
        viz.interactive_player(show_video=kwargs.get('show_video', False),
                              playback_speed=playback_speed)
    elif mode == 'frame':
        viz.show_frame(kwargs.get('frame_idx', 0), 
                      show_video=kwargs.get('show_video', False))
    elif mode == 'trajectory':
        viz.show_trajectory(
            kwargs.get('joint_idx', 0),
            kwargs.get('start_frame', 0),
            kwargs.get('end_frame', None),
            show_video=kwargs.get('show_video', False)
        )
    elif mode == 'save':
        viz.save_animation(
            kwargs.get('output_path', 'skeleton_comparison.mp4'),
            kwargs.get('start_frame', 0),
            kwargs.get('end_frame', None)
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return viz

def main():
    
    # Load data
    video_path = "/mnt/myd/IN-PERSON EXPERIMENT RECORDINGS/3HYPER FREEPLAY/3HYPER DV FREEPLAY/3HYPER.040 FREEPLAY CROPPED DV.mp4"
    keypoints_path = "/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints/3HYPER.040 FREEPLAY DV EXTRACTED 2D Keypoints.mat"
    keypoints_mat = sio.loadmat(keypoints_path)

    # Get raw data for person 0
    raw_data = keypoints_mat['person_0_2d']  # (n_joints, 2, n_frames)
    
    # Replace zeros with NaN
    raw_data = raw_data.astype(float)
    raw_data[raw_data == 0] = np.nan
    
    # Apply filtering
    filtered_data = raw_data.copy()
    fs = 30
    
    # Filter all joints (different parameter sets for each group)
    infant_group_filter_params = [
        {'cutoff': 3},  # Group 1
        {'cutoff': 3},  # Group 2
        {'cutoff': 3},  # Group 3
        {'cutoff': 3}   # Group 4
    ]
    
    parent_group_filter_params = [
        {'cutoff': 3},
        {'cutoff': 3},
        {'cutoff': 3},
        {'cutoff': 3}
    ]
       
    for group, params in zip(JOINT_GROUPS, infant_group_filter_params):
        for joint in group:
            for coord in range(2):
                sig = linear_interp(raw_data[joint, coord, :])
                cutoff = params['cutoff']
                if not np.all(np.isnan(sig)):
                    [b ,a] = signal.butter(4, cutoff, btype='lowpass', analog=False, fs=fs)
                    filtered_data[joint, coord, :] = signal.filtfilt(b, a, sig)
                        
    
    print("\n" + "="*70)
    print("SKELETAL VISUALIZATION OPTIONS")
    print("="*70)
    print("\n1. Interactive player WITH video background")
    print("2. Interactive player WITHOUT video (skeleton only)")
    print("3. Single frame WITH video background")
    print("4. Joint trajectory visualization")
    print("5. Save animation to file")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        # Interactive player with video background
        speed = float(input("Enter playback speed (0.5=slow, 1.0=normal, 2.0=fast): ") or "2.0")
        visualize_filtering_result(raw_data, filtered_data, fs, 
                                   mode='interactive', 
                                   video_path=video_path,
                                   show_video=True,
                                   playback_speed=speed)
    
    elif choice == '2':
        # Interactive player without video (skeleton only)
        speed = float(input("Enter playback speed (0.5=slow, 1.0=normal, 2.0=fast): ") or "2.0")
        visualize_filtering_result(raw_data, filtered_data, fs,
                                   mode='interactive',
                                   video_path=None,
                                   show_video=False,
                                   playback_speed=speed)
    
    elif choice == '3':
        # Single frame with video background
        frame_idx = int(input("Enter frame index (0-{}): ".format(raw_data.shape[2]-1)))
        visualize_filtering_result(raw_data, filtered_data, fs, 
                                   mode='frame', 
                                   frame_idx=frame_idx,
                                   video_path=video_path,
                                   show_video=True)
    
    elif choice == '4':
        # Joint trajectory
        joint_idx = int(input("Enter joint index (0-23): "))
        start_frame = int(input("Enter start frame: "))
        end_frame = int(input("Enter end frame: "))
        visualize_filtering_result(raw_data, filtered_data, fs,
                                   mode='trajectory', 
                                   joint_idx=joint_idx,
                                   start_frame=start_frame, 
                                   end_frame=end_frame,
                                   video_path=None,
                                   show_video=False)
    
    elif choice == '5':
        # Save animation
        output_path = input("Enter output path (e.g., comparison.mp4): ")
        start_frame = int(input("Enter start frame: "))
        end_frame = int(input("Enter end frame: "))
        visualize_filtering_result(raw_data, filtered_data, fs,
                                   mode='save', 
                                   output_path=output_path,
                                   start_frame=start_frame, 
                                   end_frame=end_frame)
    
    else:
        print("Invalid choice. Running interactive player with video.")
        visualize_filtering_result(raw_data, filtered_data, fs, 
                                   mode='interactive', 
                                   video_path=video_path,
                                   show_video=True)


if __name__ == "__main__":
    main()