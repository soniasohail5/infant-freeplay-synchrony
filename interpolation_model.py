import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from missing_gaps_stats import import_data, get_video_name, get_dyad_number
from interpolation_threshold_plotting import find_interpolated_gaps_by_gap_size
from signal_postprocessing import replace_missing
from signal_plotting import find_missing_segments_indices
from interpolation_model_functions import calculate_num_interpolated_gaps, calculate_largest_gap_not_interpolated, calculate_remaining_duration_pct, calculate_remaining_interpolated_pct, calculate_remaining_duration_seconds

'''
Models the performance of thresholded linear interpolation by extracting and analyzing the following metrics:

(1) Number of gaps filled per video (achieved by plotting a stacked bar graph which shows the number of gaps filled by the threshold over the total number of gaps)
(2) Total duration of the video preserved 
(3) From (2), the percentage of the video that was interpolated over (allows us to evaluate how likely it is that the data is actually meaningful, done in frames)
(4) Largest gap not covered by the thresholded interpolation

Provides some clues as to what dyads can be excluded from the dataset as a result of not having enough data to obtain meaningful results from
'''

GAP_THRESHOLD_FRAMES = 12

def main():
    folder_path = '/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints/'
    
    # Initialize empty summary sheets for each subject
    infant_interpolation_summary = {
        "dyad_number": [],
        "num_total_gaps": [],
        "num_gaps_accepted": [],
        "num_gaps_rejected": [],
        "remaining_duration_pct": [],
        "remaining_duration_seconds": [],
        "largest_gap_rejected": [],
        "remaining_interpolated_pct": []
    }
    
    parent_interpolation_summary =  {
        "dyad_number": [],
        "num_total_gaps": [],
        "num_gaps_accepted": [],
        "num_gaps_rejected": [],
        "remaining_duration_pct": [],
        "remaining_duration_seconds": [],
        "largest_gap_rejected": [],
        "remaining_interpolated_pct": []
    }

    # Extract metrics from each dyad in the dataset
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        dyad_info = import_data(full_path)
        infant_keypoint_data = np.array(dyad_info["infant"])
        parent_keypoint_data = np.array(dyad_info["parent"])
        total_video_duration = parent_keypoint_data.shape[2]
        dyad_name = get_video_name(full_path)
        dyad_number = get_dyad_number(dyad_name)
        
        print(f"Extracting from {file} .....")
        
        infant_signal, infant_nan = replace_missing(infant_keypoint_data[15, 0, :])
        parent_signal, parent_nan = replace_missing(parent_keypoint_data[15, 0, :])
        
        infant_total_gaps = find_missing_segments_indices(infant_signal)
        parent_total_gaps = find_missing_segments_indices(parent_signal)
        
        infant_total_gaps_num = len(infant_total_gaps)
        parent_total_gaps_num = len(parent_total_gaps)
        
        infant_accepted_gaps = find_interpolated_gaps_by_gap_size(infant_signal, GAP_THRESHOLD_FRAMES)
        parent_accepted_gaps = find_interpolated_gaps_by_gap_size(parent_signal, GAP_THRESHOLD_FRAMES)
    
        infant_accepted_gaps_num = calculate_num_interpolated_gaps(infant_signal)
        parent_accepted_gaps_num = calculate_num_interpolated_gaps(parent_signal)
        
        infant_rejected_gaps = infant_total_gaps_num - infant_accepted_gaps_num
        parent_rejected_gaps = parent_total_gaps_num - parent_accepted_gaps_num
        
        infant_remaining_duration_seconds = calculate_remaining_duration_seconds(infant_signal, total_video_duration)
        parent_remaining_duration_seconds = calculate_remaining_duration_seconds(parent_signal, total_video_duration)
        
        infant_remaining_duration_pct = calculate_remaining_duration_pct(infant_signal, total_video_duration)
        parent_remaining_duration_pct = calculate_remaining_duration_pct(parent_signal, total_video_duration)
    
        infant_remaining_interpolated_pct = calculate_remaining_interpolated_pct(infant_signal, infant_accepted_gaps, total_video_duration)
        parent_remaining_interpolated_pct = calculate_remaining_interpolated_pct(parent_signal, parent_accepted_gaps, total_video_duration)
    
        infant_largest_rejected_gap = calculate_largest_gap_not_interpolated(infant_signal)
        parent_largest_rejected_gap = calculate_largest_gap_not_interpolated(parent_signal)
        
        infant_interpolation_summary["dyad_number"].append(dyad_number)
        infant_interpolation_summary["num_total_gaps"].append(infant_total_gaps_num)
        infant_interpolation_summary["num_gaps_accepted"].append(infant_accepted_gaps_num)
        infant_interpolation_summary["num_gaps_rejected"].append(infant_rejected_gaps)
        infant_interpolation_summary["largest_gap_rejected"].append(infant_largest_rejected_gap)
        infant_interpolation_summary["remaining_duration_seconds"].append(infant_remaining_duration_seconds)
        infant_interpolation_summary["remaining_duration_pct"].append(infant_remaining_duration_pct)
        infant_interpolation_summary["remaining_interpolated_pct"].append(infant_remaining_interpolated_pct)
        
        parent_interpolation_summary["dyad_number"].append(dyad_number)
        parent_interpolation_summary["num_total_gaps"].append(parent_total_gaps_num)
        parent_interpolation_summary["num_gaps_accepted"].append(parent_accepted_gaps_num)
        parent_interpolation_summary["num_gaps_rejected"].append(parent_rejected_gaps)
        parent_interpolation_summary["remaining_duration_pct"].append(parent_remaining_duration_pct)
        parent_interpolation_summary["remaining_duration_seconds"].append(parent_remaining_duration_seconds)
        parent_interpolation_summary["largest_gap_rejected"].append(parent_largest_rejected_gap)
        parent_interpolation_summary["remaining_interpolated_pct"].append(parent_remaining_interpolated_pct)

    # Combine all dyads into single DataFrame  
    infant_interpolation_per_video_df = pd.DataFrame(infant_interpolation_summary)
    parent_interpolation_per_video_df = pd.DataFrame(parent_interpolation_summary)

    # Write dataframes into an excel sheet 
    with pd.ExcelWriter('interpolation_model_summary.xlsx') as writer:
        infant_interpolation_per_video_df.to_excel(writer, sheet_name='Infant', index=False)
        parent_interpolation_per_video_df.to_excel(writer, sheet_name='Parent', index=False)

if __name__ == "__main__":    
    main()
