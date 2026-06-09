"""
Multi-Person Skeletal Visualization Tool for Pose Estimation Data
Overlay layout showing multiple people on the same view
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Slider, Button
import cv2
import scipy.io as sio
from scipy import signal
from signal_postprocessing import linear_interp, spline_interp, movmad_filter

plt.rcParams['animation.ffmpeg_path'] = '/mnt/c/ffmpeg/bin/ffmpeg.exe'

# ============== JOINT CONFIGURATION ==============

JOINT_EDGES = [
    # Core/spine connections
    (1, 4), (1, 0), (2, 5), (2, 0), (3, 6), (3, 0),
    # Leg connections
    (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
    # Hip to shoulders/head
    (9, 13), (9, 14),
    # Arm connections
    (12, 15), (13, 16), (14, 17),
    (16, 18), (17, 19), (18, 20), (19, 21), (20, 22), (21, 23),
]

# Per-person colors for raw and filtered views
PERSON_COLORS_RAW      = ['#00FF00', '#00FFFF']   # green, cyan
PERSON_COLORS_FILTERED = ['#0080FF', '#FF0080']   # blue, pink


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
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Loaded video: {self.total_frames} frames, {self.fps} fps, "
              f"{self.width}x{self.height}")

    def get_frame(self, frame_idx):
        """Get a specific frame (with caching)"""
        if frame_idx in self.frames_cache:
            return self.frames_cache[frame_idx]

        if self.cap is None:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames_cache[frame_idx] = frame_rgb
            return frame_rgb
        return None

    def __del__(self):
        if self.cap is not None:
            self.cap.release()


# ============== FILTERING ==============

def apply_butterworth_to_all_keypoints(raw_data, cutoff, order=4, fs=30):
    """
    Apply Butterworth low-pass filter to ALL keypoints uniformly.

    Args:
        raw_data: (n_joints, 2, n_frames) array
        cutoff: Cutoff frequency in Hz
        order: Filter order (default 4)
        fs: Sampling frequency

    Returns:
        filtered_data: Same shape as raw_data
    """
    filtered_data = raw_data.copy()
    b, a = signal.butter(order, cutoff, btype='low', fs=fs)

    for joint in range(raw_data.shape[0]):
        for coord in range(2):
            sig = raw_data[joint, coord, :]
            if not np.all(np.isnan(sig)):
                sig_filled = np.nan_to_num(sig, nan=np.nanmean(sig))
                filtered_data[joint, coord, :] = signal.filtfilt(b, a, sig_filled)

    return filtered_data

def apply_medfilt_to_all_keypoints(raw_data, window_size):
    """
    Apply Butterworth low-pass filter to ALL keypoints uniformly.

    Args:
        raw_data: (n_joints, 2, n_frames) array
        cutoff: Cutoff frequency in Hz
        order: Filter order (default 4)
        fs: Sampling frequency

    Returns:
        filtered_data: Same shape as raw_data
    """
    filtered_data = raw_data.copy()

    for joint in range(raw_data.shape[0]):
        for coord in range(2):
            sig = raw_data[joint, coord, :]
            if not np.all(np.isnan(sig)):
                sig_filled = np.nan_to_num(sig, nan=0)
                filtered_data[joint, coord, :] = movmad_filter(sig_filled, window_size)
                
    return filtered_data

def apply_linear_interp(raw_data):
    filtered_data = raw_data.copy()
    for joint in range(raw_data.shape[0]):
        for coord in range(2):
            sig = raw_data[joint, coord, :]
            if np.any(np.isnan(sig)):
                linear_interp_data = linear_interp(sig)
                filtered_data[joint, coord, :] = linear_interp_data
        
    return filtered_data

def apply_spline_interp(raw_data):
    filtered_data = raw_data.copy()
    
    for joint in range(raw_data.shape[0]):
        for coord in range(2):
            sig = raw_data[joint, coord, :]
            if np.any(np.isnan(sig)):
                spline_interp_data = spline_interp(sig)
                filtered_data[joint, coord, :] = spline_interp_data
    
    return filtered_data


# ============== MULTI-PERSON VISUALIZER ==============

class MultiPersonSkeletalVisualizer:
    """
    Visualize multiple people (e.g., infant and parent) in overlay layout.
    Left panel  : raw skeletons for all people.
    Right panel : filtered skeletons (color) with raw skeletons underneath (gray).
    """

    def __init__(self, persons_raw, persons_filtered, person_labels,
                 fs=30, title="Multi-Person Skeletal Comparison", video_path=None):
        """
        Args:
            persons_raw:      List of raw keypoint arrays [(n_joints, 2, n_frames), ...]
            persons_filtered: List of filtered keypoint arrays (same shape)
            person_labels:    List of string labels, e.g. ['Infant', 'Parent']
            fs:               Sampling frequency (Hz)
            title:            Figure title
            video_path:       Optional path to source video for background overlay
        """
        self.persons_raw      = [np.array(p, dtype=float) for p in persons_raw]
        self.persons_filtered = [np.array(p, dtype=float) for p in persons_filtered]
        self.person_labels    = person_labels
        self.n_persons        = len(persons_raw)
        self.fs               = fs
        self.title            = title

        self.n_joints  = self.persons_raw[0].shape[0]
        self.n_frames  = self.persons_raw[0].shape[2]
        self.current_frame = 0
        self.playing   = False
        self.anim      = None

        # Load video if provided
        self.video_loader = VideoLoader(video_path) if video_path else None

        # Compute display bounds from all data
        all_data = self.persons_raw + self.persons_filtered
        all_concat = np.concatenate(all_data, axis=2)
        valid_x = all_concat[:, 0, :][~np.isnan(all_concat[:, 0, :])]
        valid_y = all_concat[:, 1, :][~np.isnan(all_concat[:, 1, :])]

        if self.video_loader:
            self.xlim = (0, self.video_loader.width)
            self.ylim = (0, self.video_loader.height)
        else:
            margin = 50
            self.xlim = (np.min(valid_x) - margin, np.max(valid_x) + margin)
            self.ylim = (np.min(valid_y) - margin, np.max(valid_y) + margin)

    # ------------------------------------------------------------------
    # Core drawing helper
    # ------------------------------------------------------------------

    def _draw_skeleton(self, ax, data, frame, color, alpha=1.0, label=None):
        """
        Draw skeleton for a single person at a given frame.

        Args:
            ax:     Matplotlib axes
            data:   (n_joints, 2, n_frames) array
            frame:  Frame index
            color:  Uniform color string for all joints/edges
            alpha:  Transparency
            label:  Legend label (optional)
        """
        x = data[:, 0, frame]
        y = data[:, 1, frame]

        # Draw edges
        for i_start, i_end in JOINT_EDGES:
            if i_start < len(x) and i_end < len(x):
                if not (np.isnan(x[i_start]) or np.isnan(x[i_end])):
                    ax.plot(
                        [x[i_start], x[i_end]],
                        [y[i_start], y[i_end]],
                        color=color, linewidth=2, alpha=alpha, zorder=1
                    )

        # Draw joints
        valid = ~np.isnan(x)
        if valid.any():
            ax.scatter(x[valid], y[valid], c=color, s=50, alpha=alpha, zorder=5)

        # Legend proxy
        if label is not None:
            ax.scatter([], [], c=color, s=100, label=label)

    def _draw_video_bg(self, axes, frame):
        """Draw video frame as background on all provided axes."""
        if self.video_loader:
            video_frame = self.video_loader.get_frame(frame)
            if video_frame is not None:
                extent = [0, self.video_loader.width, self.video_loader.height, 0]
                for ax in axes:
                    ax.imshow(video_frame, extent=extent, aspect='auto', zorder=0)

    def _setup_axes(self, axes, titles):
        for ax, title in zip(axes, titles):
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_title(title, fontsize=12)

    # ------------------------------------------------------------------
    # Display update (used by interactive player)
    # ------------------------------------------------------------------

    def _update_display(self, axes, frame, show_video=False):
        """Redraw both panels for the given frame."""
        for ax in axes:
            ax.clear()

        self._setup_axes(axes, ['Raw (All People)', 'Filtered (with Raw in Gray)'])

        if show_video:
            self._draw_video_bg(axes, frame)

        # Left panel – raw skeletons
        for i in range(self.n_persons):
            color = PERSON_COLORS_RAW[i % len(PERSON_COLORS_RAW)]
            self._draw_skeleton(axes[0], self.persons_raw[i], frame,
                                color=color, alpha=0.85,
                                label=self.person_labels[i])
        axes[0].legend(loc='upper right', fontsize=10)

        # Right panel – raw in gray, then filtered on top
        for i in range(self.n_persons):
            self._draw_skeleton(axes[1], self.persons_raw[i], frame,
                                color='gray', alpha=0.60)

        for i in range(self.n_persons):
            color = PERSON_COLORS_FILTERED[i % len(PERSON_COLORS_FILTERED)]
            self._draw_skeleton(axes[1], self.persons_filtered[i], frame,
                                color=color, alpha=0.90,
                                label=f'{self.person_labels[i]} (filtered)')
        axes[1].legend(loc='upper right', fontsize=10)

    # ------------------------------------------------------------------
    # Interactive player
    # ------------------------------------------------------------------

    def interactive_player(self, show_video=False, playback_speed=1.0):
        """
        Launch interactive animation window with play/pause/step controls.

        Args:
            show_video:      Overlay video background if available
            playback_speed:  Initial playback speed multiplier
        """
        if self.n_frames < 2:
            print(f"Error: Need at least 2 frames, got {self.n_frames}")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(bottom=0.25)

        title_suffix  = " (with video)" if show_video and self.video_loader else ""
        speed_suffix  = f" [{playback_speed}x speed]" if playback_speed != 1.0 else ""
        fig.suptitle(self.title + title_suffix + speed_suffix, fontsize=14)

        self._update_display(axes, 0, show_video)

        # Slider
        ax_slider = plt.axes([0.15, 0.08, 0.7, 0.02])
        slider = Slider(ax_slider, 'Frame', valmin=0,
                        valmax=max(1, self.n_frames - 1),
                        valinit=0, valstep=1)

        # Time display
        time_text = fig.text(0.5, 0.03,
                             f'Time: 0.00s | Speed: {playback_speed}x',
                             ha='center', fontsize=12)

        # Buttons
        ax_play       = plt.axes([0.15, 0.01, 0.08, 0.03])
        ax_pause      = plt.axes([0.24, 0.01, 0.08, 0.03])
        ax_reset      = plt.axes([0.33, 0.01, 0.08, 0.03])
        ax_step_back  = plt.axes([0.50, 0.01, 0.08, 0.03])
        ax_step_fwd   = plt.axes([0.59, 0.01, 0.08, 0.03])
        ax_speed_down = plt.axes([0.68, 0.01, 0.08, 0.03])
        ax_speed_up   = plt.axes([0.77, 0.01, 0.08, 0.03])

        btn_play       = Button(ax_play,       'Play')
        btn_pause      = Button(ax_pause,      'Pause')
        btn_reset      = Button(ax_reset,      'Reset')
        btn_step_back  = Button(ax_step_back,  '< Step')
        btn_step_fwd   = Button(ax_step_fwd,   'Step >')
        btn_speed_down = Button(ax_speed_down, 'Speed -')
        btn_speed_up   = Button(ax_speed_up,   'Speed +')

        current_speed = [playback_speed]

        def update_slider(val):
            frame = int(slider.val)
            self._update_display(axes, frame, show_video)
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
            slider.set_val(max(0, slider.val - 1))

        def step_fwd(event):
            slider.set_val(min(self.n_frames - 1, slider.val + 1))

        def speed_down(event):
            current_speed[0] = max(0.25, current_speed[0] - 0.25)
            time_text.set_text(f'Time: {int(slider.val)/self.fs:.2f}s | Speed: {current_speed[0]}x')
            fig.suptitle(self.title + title_suffix + f" [{current_speed[0]}x speed]")
            fig.canvas.draw_idle()
            if self.playing:
                self._animate(fig, axes, slider, time_text, show_video, current_speed)

        def speed_up(event):
            current_speed[0] = min(4.0, current_speed[0] + 0.25)
            time_text.set_text(f'Time: {int(slider.val)/self.fs:.2f}s | Speed: {current_speed[0]}x')
            fig.suptitle(self.title + title_suffix + f" [{current_speed[0]}x speed]")
            fig.canvas.draw_idle()
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

        plt.show()

    def _animate(self, fig, axes, slider, time_text, show_video, current_speed):
        """Internal FuncAnimation loop used by interactive_player."""
        def update(frame):
            if not self.playing:
                return
            current = int(slider.val)
            next_frame = (current + 1) % self.n_frames
            slider.set_val(next_frame)

        interval = (1000 / self.fs) / current_speed[0]

        if self.anim is not None:
            self.anim.event_source.stop()

        self.anim = FuncAnimation(fig, update, interval=interval,
                                  cache_frame_data=False)
        fig.canvas.draw()

    # ------------------------------------------------------------------
    # Save animation
    # ------------------------------------------------------------------

    def save_animation(self, output_path, show_video=True,
                       fps=None, start_frame=0, end_frame=None,
                       dpi=100, figsize=(14, 7)):
        """
        Render and save the two-panel animation to a video file.

        Both panels include the video background overlay when show_video=True
        and a video source was provided.  The right panel always draws the
        raw skeleton in gray beneath the filtered skeleton.

        Args:
            output_path:  Output file path, e.g. 'output.mp4'
            show_video:   Overlay video background on both panels
            fps:          Output frame rate (defaults to self.fs)
            start_frame:  First frame to render (inclusive)
            end_frame:    Last frame to render (inclusive, defaults to last)
            dpi:          Figure DPI
            figsize:      Figure size in inches
        """
        fps = fps or self.fs
        end_frame = end_frame if end_frame is not None else self.n_frames - 1
        frames_to_render = range(start_frame, end_frame + 1)
        n_render = len(frames_to_render)

        print(f"Saving animation: frames {start_frame}–{end_frame} "
              f"({n_render} frames) → '{output_path}'")

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(self.title + (" (with video)" if show_video and self.video_loader else ""),
                     fontsize=14)

        # Draw initial frame so axes are configured
        self._update_display(axes, start_frame, show_video)

        def update(frame_idx):
            self._update_display(axes, frame_idx, show_video)
            if frame_idx % 100 == 0:
                print(f"  Rendered frame {frame_idx}/{end_frame}")

        anim = FuncAnimation(fig, update,
                             frames=frames_to_render,
                             interval=1000 / fps,
                             cache_frame_data=False)

        writer = FFMpegWriter(fps=fps, metadata={'title': self.title},
                              extra_args=['-vcodec', 'libx264',
                                          '-pix_fmt', 'yuv420p'])
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Animation saved to: {output_path}")

    def save_raw_animation(self, output_path, show_video=True,
                           fps=None, start_frame=0, end_frame=None,
                           dpi=100, figsize=(8, 7)):
        """
        Save a single-panel animation of the raw skeletons only.

        Args:
            output_path:  Output file path
            show_video:   Overlay video background
            fps:          Output frame rate (defaults to self.fs)
            start_frame:  First frame to render
            end_frame:    Last frame to render (inclusive)
            dpi:          Figure DPI
            figsize:      Figure size in inches
        """
        fps = fps or self.fs
        end_frame = end_frame if end_frame is not None else self.n_frames - 1
        frames_to_render = range(start_frame, end_frame + 1)

        print(f"Saving raw animation → '{output_path}'")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle(self.title + " – Raw", fontsize=14)

        def update(frame_idx):
            ax.clear()
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_title('Raw (All People)', fontsize=12)

            if show_video and self.video_loader:
                video_frame = self.video_loader.get_frame(frame_idx)
                if video_frame is not None:
                    ax.imshow(video_frame,
                              extent=[0, self.video_loader.width,
                                      self.video_loader.height, 0],
                              aspect='auto', zorder=0)

            for i in range(self.n_persons):
                color = PERSON_COLORS_RAW[i % len(PERSON_COLORS_RAW)]
                self._draw_skeleton(ax, self.persons_raw[i], frame_idx,
                                    color=color, alpha=0.85,
                                    label=self.person_labels[i])
            ax.legend(loc='upper right', fontsize=10)

            if frame_idx % 100 == 0:
                print(f"  Rendered frame {frame_idx}/{end_frame}")

        anim = FuncAnimation(fig, update,
                             frames=frames_to_render,
                             interval=1000 / fps,
                             cache_frame_data=False)

        writer = FFMpegWriter(fps=fps, metadata={'title': self.title + ' Raw'},
                              extra_args=['-vcodec', 'libx264',
                                          '-pix_fmt', 'yuv420p'])
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Raw animation saved to: {output_path}")

    def save_filtered_animation(self, output_path, show_video=True,
                                fps=None, start_frame=0, end_frame=None,
                                dpi=100, figsize=(8, 7)):
        """
        Save a single-panel animation of the filtered skeletons,
        with the original raw skeleton drawn underneath in gray.

        Args:
            output_path:  Output file path
            show_video:   Overlay video background
            fps:          Output frame rate (defaults to self.fs)
            start_frame:  First frame to render
            end_frame:    Last frame to render (inclusive)
            dpi:          Figure DPI
            figsize:      Figure size in inches
        """
        fps = fps or self.fs
        end_frame = end_frame if end_frame is not None else self.n_frames - 1
        frames_to_render = range(start_frame, end_frame + 1)

        print(f"Saving filtered animation → '{output_path}'")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.suptitle(self.title + " – Filtered (gray = raw)", fontsize=14)

        def update(frame_idx):
            ax.clear()
            ax.set_xlim(self.xlim)
            ax.set_ylim(self.ylim)
            ax.invert_yaxis()
            ax.set_aspect('equal')
            ax.set_title('Filtered (with Raw in Gray)', fontsize=12)

            if show_video and self.video_loader:
                video_frame = self.video_loader.get_frame(frame_idx)
                if video_frame is not None:
                    ax.imshow(video_frame,
                              extent=[0, self.video_loader.width,
                                      self.video_loader.height, 0],
                              aspect='auto', zorder=0)

            # Raw in gray underneath
            for i in range(self.n_persons):
                self._draw_skeleton(ax, self.persons_raw[i], frame_idx,
                                    color='gray', alpha=0.60)

            # Filtered on top in color
            for i in range(self.n_persons):
                color = PERSON_COLORS_FILTERED[i % len(PERSON_COLORS_FILTERED)]
                self._draw_skeleton(ax, self.persons_filtered[i], frame_idx,
                                    color=color, alpha=0.90,
                                    label=f'{self.person_labels[i]} (filtered)')
            ax.legend(loc='upper right', fontsize=10)

            if frame_idx % 100 == 0:
                print(f"  Rendered frame {frame_idx}/{end_frame}")

        anim = FuncAnimation(fig, update,
                             frames=frames_to_render,
                             interval=1000 / fps,
                             cache_frame_data=False)

        writer = FFMpegWriter(fps=fps, metadata={'title': self.title + ' Filtered'},
                              extra_args=['-vcodec', 'libx264',
                                          '-pix_fmt', 'yuv420p'])
        anim.save(output_path, writer=writer, dpi=dpi)
        plt.close(fig)
        print(f"Filtered animation saved to: {output_path}")


# ============== MAIN ==============

def main():
    """Multi-person visualization with overlay layout"""

    # Load data
    keypoints_path = (
        "/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints/"
        "3HYPER.025 FREEPLAY DV EXTRACTED 2D Keypoints.mat"
    )
    keypoints_mat = sio.loadmat(keypoints_path)
    video_path = "/mnt/e/IN-PERSON EXPERIMENT RECORDINGS/3HYPER FREEPLAY/3HYPER DV FREEPLAY/3HYPER.025 FREEPLAY DV.mp4" # fix video path 

    person_0_raw = keypoints_mat['person_0_2d'].astype(float)
    person_1_raw = keypoints_mat['person_1_2d'].astype(float)

    person_0_raw[person_0_raw == 0] = np.nan
    person_1_raw[person_1_raw == 0] = np.nan

    print("\n" + "=" * 70)
    print("MULTI-PERSON SKELETAL VISUALIZATION")
    print("=" * 70)

    # Filtering parameters
    fs = 30
    
    '''
    cutoff_freq  = float(input("\nEnter Butterworth cutoff frequency (Hz, default 6): ") or "6")
    filter_order = int(input("Enter filter order (default 4): ") or "4")

    print(f"\nApplying {filter_order}th-order Butterworth filter at {cutoff_freq} Hz...")
    print("  Filtering Person 0 (Infant)...")
    '''
    # 1: Linear Interpolation only 
    person_0_interpolated = apply_medfilt_to_all_keypoints(apply_linear_interp(person_0_raw), 30)
    
    # 2: Linear Interpolation + Butterworth Filter
    # person_0_filtered = apply_butterworth_to_all_keypoints(person_0_interpolated, cutoff_freq, order=filter_order, fs=fs)
    
    # 3: Spline Interpolation only
    # person_0_interpolated = apply_spline_interp(person_0_raw)
    
    # 4: Spline Interpolation + Butterworth Filter
    # person_0_filtered = apply_butterworth_to_all_keypoints(person_0_interpolated, cutoff_freq, order=filter_order, fs=fs)

    print("  Filtering Person 1 (Parent)...")
    # 1: Linear Interpolation only 
    person_1_interpolated = apply_medfilt_to_all_keypoints(apply_linear_interp(person_1_raw), 30)

    # 2: Linear Interpolation + Butterworth Filter
    # person_1_filtered = apply_butterworth_to_all_keypoints(person_1_interpolated, cutoff_freq, order=filter_order, fs=fs)
    
    # 3: Spline Interpolation only 
    # person_1_interpolated = apply_spline_interp(person_1_raw)
    
    # 4: Spline Interpolation + Butterworth Filter 
    # person_1_filtered = apply_butterworth_to_all_keypoints(person_1_interpolated, cutoff_freq, order=filter_order, fs=fs)

    print("Filtering complete!")

    # Visualisation settings
    speed = float(input("\nEnter playback speed (0.5=slow, 1.0=normal, 2.0=fast, default 2.0): ") or "2.0")

    show_video_input = input("Show video background? (y/n, default y): ").strip().lower()
    show_video = show_video_input != 'n'

    # Create visualizer
    viz = MultiPersonSkeletalVisualizer(
        persons_raw=[person_0_raw, person_1_raw],
        persons_filtered=[person_0_interpolated, person_1_interpolated],
        person_labels=['Infant', 'Parent'],
        fs=fs,
        title="Infant-Parent Interaction",
        video_path=video_path if show_video else None,
    )

    # Ask whether to save or view interactively
    save_input = input("\nSave animation to file? (y/n, default n): ").strip().lower()
    print("\nSave options:")
    print("  1 – Both panels (raw + filtered)")
    print("  2 – Raw only")
    print("  3 – Filtered only")
    save_choice = input("Choose (default 1): ").strip() or "1"

    out_base = input("Output filename base (default 'skeleton_animation'): ").strip() \
                or "skeleton_animation"

    if save_input == "1":
        if save_choice == "1":
            viz.save_animation(f"{out_base}.mp4", show_video=show_video)
        elif save_choice == "2":
            viz.save_raw_animation(f"{out_base}_raw.mp4", show_video=show_video)
        elif save_choice == "3":
            viz.save_filtered_animation(f"{out_base}_filtered.mp4", show_video=show_video)
        else:
            print("Invalid choice, skipping save.")
            
    else:
        print("\nLaunching interactive player...")
        viz.interactive_player(show_video=show_video, playback_speed=speed)


if __name__ == "__main__":
    main()