import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from missing_gaps_stats import import_data, get_video_name, get_dyad_number
from interpolation_threshold import find_gap_indices
from interpolation_threshold_plotting import find_interpolated_gaps_by_gap_size

'''
Models the performance of thresholded linear interpolation by extracting and analyzing the following metrics:

(1) Number of gaps filled per video (achieved by plotting a stacked bar graph which shows the number of gaps filled by the threshold over the total number of gaps)
(2) Total duration of the video preserved 
(3) From (2), the percentage of the video that was interpolated over (allows us to evaluate how likely it is that the data is actually meaningful, done in frames)
(4) Largest gap not covered by the thresholded interpolation

Provides some clues as to what dyads can be excluded from the dataset as a result of not having enough data to obtain meaningful results from
'''
GAP_THRESHOLD_FRAMES = 12
GAP_THRESHOLD_SECONDS = 0.4

def calculate_num_interpolated_gaps(keypoint_data, gap_threshold):
    # Returns the number of gaps that would be filled by linear interpolation under the specified gap size threshold
    valid_gaps = find_interpolated_gaps_by_gap_size(keypoint_data, gap_threshold)
    return valid_gaps

def plot_interpolated_gaps_per_video(num_gaps_per_video, total_gaps_per_video):
    # Plots a stacked bar graph showing the number of gaps filled by interpolation vs. total gaps for each video

    videos = range(len(num_gaps_per_video))
    plt.bar(videos, num_gaps_per_video, label='Interpolated Gaps')
    plt.bar(videos, total_gaps_per_video, bottom=num_gaps_per_video, label='Total Gaps', alpha=0.5)
    plt.xlabel('Video Index')
    plt.ylabel('Number of Gaps')
    plt.title('Number of Interpolated Gaps vs Total Gaps per Video')
    plt.legend()
    plt.show()
    
def calculate_total_interpolated_duration(accepted_gaps_in_video):
    # Returns the total duration of the video preserved by interpolation under the specified gap size threshold
    total_interpolated_duration = sum([len(gap) for gap in accepted_gaps_in_video])
    return total_interpolated_duration

def calculate_percentage_video_interpolated(num_accepted_gaps, total_video_duration):
    # Returns the percentage of the video that was interpolated over under the specified gap size threshold
    total_interpolated_duration = calculate_total_interpolated_duration(num_accepted_gaps)
    percentage_interpolated = (total_interpolated_duration / total_video_duration) * 100
    return percentage_interpolated

def calculate_largest_gap_not_interpolated(keypoint_data, gap_threshold):
    # Returns the size of the largest gap that would not be filled by linear interpolation under the specified gap size threshold
    valid_gaps = find_interpolated_gaps_by_gap_size(keypoint_data, gap_threshold)
    if not valid_gaps:
        return 0
    largest_gap_not_interpolated = max(len(gap) for gap in valid_gaps)
    return largest_gap_not_interpolated

def plot_largest_gap_not_interpolated(largest_gaps_per_video):
    # Plots a histogram of the largest gap not interpolated for each video in the dataset under the specified gap size threshold
    import matplotlib.pyplot as plt

    plt.hist(largest_gaps_per_video, bins=20, edgecolor='black')
    plt.xlabel('Largest Gap Not Interpolated (frames)')
    plt.ylabel('Number of Videos')
    plt.title('Distribution of Largest Gaps Not Interpolated per Video')
    plt.show()

def main():
    folder_path = '/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints/'
    
    infant_interpolation_summary = {
        "dyad_number": [],
        "total_gaps": [],
        "num_gaps_accepted": [],
        "num_gaps_rejected": [],
        "remaining_duration_pct": [],
        "largest_gap_rejected": [],
    }
    
    parent_interpolation_summary =  {
        "dyad_number": [],
        "total_gaps": [],
        "num_gaps_accepted": [],
        "num_gaps_rejected": [],
        "remaining_duration_pct": [],
        "largest_gap_rejected": [],
    }

    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        dyad_info = import_data(full_path)
        infant_keypoint_data = np.array(dyad_info["infant"])
        parent_keypoint_data = np.array(dyad_info["parent"])
        total_video_duration = parent_keypoint_data.shape[2]
        dyad_name = get_video_name(full_path)
        dyad_number = get_dyad_number(dyad_name)
        
        infant_signal = infant_keypoint_data[0, 0, :]
        parent_signal = parent_keypoint_data[0, 0, :]
        
        infant_total_gaps = find_gap_indices(infant_signal)
        parent_total_gaps = find_gap_indices(parent_signal)
        
        infant_accepted_gaps = calculate_num_interpolated_gaps(infant_signal, GAP_THRESHOLD_FRAMES)
        parent_accepted_gaps = calculate_num_interpolated_gaps(parent_signal, GAP_THRESHOLD_FRAMES)
        
        infant_rejected_gaps = len(infant_total_gaps) - len(infant_accepted_gaps)
        parent_rejected_gaps = len(parent_total_gaps) - len(parent_accepted_gaps)
        
        infant_interpolated_duration_pct = calculate_percentage_video_interpolated(infant_accepted_gaps, total_video_duration)
        parent_interpolated_duration_pct = calculate_percentage_video_interpolated(parent_accepted_gaps, total_video_duration)
        
        infant_largest_rejected_gap = calculate_largest_gap_not_interpolated(infant_signal, GAP_THRESHOLD_FRAMES)
        parent_largest_rejected_gap = calculate_largest_gap_not_interpolated(parent_signal, GAP_THRESHOLD_FRAMES)
        
        infant_interpolation_summary["dyad_number"].append(dyad_number)
        infant_interpolation_summary["total_gaps"].append(infant_total_gaps)
        infant_interpolation_summary["num_gaps_accepted"].append(infant_accepted_gaps)
        infant_interpolation_summary["num_gaps_rejected"].append(infant_rejected_gaps)
        infant_interpolation_summary["remaining_duration_pct"].append(infant_interpolated_duration_pct)
        infant_interpolation_summary["largest_gap_rejected"].append(infant_largest_rejected_gap)
        
        parent_interpolation_summary["dyad_number"].append(dyad_number)
        parent_interpolation_summary["total_gaps"].append(parent_total_gaps)
        parent_interpolation_summary["num_gaps_accepted"].append(parent_accepted_gaps)
        parent_interpolation_summary["num_gaps_rejected"].append(parent_rejected_gaps)
        parent_interpolation_summary["remaining_duration_pct"].append(parent_interpolated_duration_pct)
        parent_interpolation_summary["largest_gap_rejected"].append(parent_largest_rejected_gap)
        
    infant_interpolation_per_video_df = pd.DataFrame(infant_interpolation_summary)
    parent_interpolation_per_video_df = pd.DataFrame(parent_interpolation_summary)

if __name__ == "__main__":    
    main()

