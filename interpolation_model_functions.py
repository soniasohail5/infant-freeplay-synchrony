import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from missing_gaps_stats import import_data, get_video_name, get_dyad_number
from interpolation_threshold_plotting import find_interpolated_gaps_by_gap_size
from signal_postprocessing import replace_missing
from signal_plotting import find_missing_segments_indices

GAP_THRESHOLD_FRAMES = 12
GAP_THRESHOLD_SECONDS = 0.4
VIDEO_DURATION = 240

def calculate_num_interpolated_gaps(keypoint_data):
    # Returns the number of gaps that would be filled by linear interpolation under the specified gap size threshold
    valid_gaps = find_interpolated_gaps_by_gap_size(keypoint_data, GAP_THRESHOLD_FRAMES)
    return len(valid_gaps)

def find_rejected_gaps(keypoint_data):
    total_gaps = find_missing_segments_indices(keypoint_data)
    rejected_gaps = [gap for gap in total_gaps if len(gap) > GAP_THRESHOLD_FRAMES]
    return rejected_gaps

def calculate_total_interpolated_duration(accepted_gaps_in_video):
    # Returns the total duration of the video preserved by interpolation under the specified gap size threshold
    total_interpolated_duration = sum([len(gap) for gap in accepted_gaps_in_video])
    return total_interpolated_duration

def calculate_percentage_video_interpolated(num_accepted_gaps, total_video_duration):
    # Returns the percentage of the video that was interpolated over under the specified gap size threshold
    total_interpolated_duration = calculate_total_interpolated_duration(num_accepted_gaps)
    percentage_interpolated = (total_interpolated_duration / total_video_duration) * 100

    return percentage_interpolated

def calculate_largest_gap_not_interpolated(keypoint_data):
    # Returns the size of the largest gap that would not be filled by linear interpolation under the specified gap size threshold
    candidate_gaps = find_rejected_gaps(keypoint_data)
    if len(candidate_gaps) == 0:
        return 0
    
    len_gaps = [len(gap) for gap in candidate_gaps]
    return max(len_gaps)

def calculate_remaining_duration(keypoint_data, total_video_duration):
     rejected_gaps = find_rejected_gaps(keypoint_data)
     total_rejected_frames = sum([len(gap) for gap in rejected_gaps])
    
     remaining_duration_frames = total_video_duration - total_rejected_frames
     
     return remaining_duration_frames
    
def calculate_remaining_duration_pct(keypoint_data, total_video_duration):
    remaining_duration_frames = calculate_remaining_duration(keypoint_data, total_video_duration)
    percentage_remaining = (remaining_duration_frames/total_video_duration) * 100
    
    return percentage_remaining

def calculate_remaining_interpolated_pct(keypoint_data, accepted_gaps, total_video_duration):
    total_interpolated_frames = calculate_total_interpolated_duration(accepted_gaps)
    remaining_duration_frames = calculate_remaining_duration(keypoint_data, total_video_duration)
    
    percentage_remaining_interpolated = (total_interpolated_frames/remaining_duration_frames) * 100
    
    return percentage_remaining_interpolated
    
def frames_to_seconds(quantity_in_frames, total_video_duration):
    # Assumes all videos are exactly 4 minutes long
    fps = total_video_duration/VIDEO_DURATION
    quantity_in_seconds = quantity_in_frames/fps
    
    return quantity_in_seconds

def calculate_remaining_duration_seconds(keypoint_data, total_video_duration):
    remaining_duration_frames = calculate_remaining_duration(keypoint_data, total_video_duration)
    remaining_duration_seconds = frames_to_seconds(remaining_duration_frames, total_video_duration)
    
    return remaining_duration_seconds
    
    

