import os
import re
import math
import numpy as np
import pandas as pd
import scipy.io as sio
from signal_plotting import find_missing_segments_indices
from signal_postprocessing import replace_missing

VIDEO_DURATION = 240
JOINT_NAMES = ["Head"]
JOINT_INDICES = [16]
CLUSTER_THRESHOLD_FRAMES = 30 

'''
Generates an .xlsx file that lists several characteristics from several joint keypoints in each video in the dataset: 
    (1) video duration (string), 
    (2) number of segments (int), 
    (3) duration of each segment (array of ints)
    
All characteristics will have two columns - one measured in frames (int), and one measured in minutes:seconds (string)
Done for each desired joint keypoint (head, l/r shoulders, l/r elbows, l/r wrists, spines 2 & 3)

EDIT: 3/15/2026
An additional sheet is made for clusters for each subject (several small missing data segments are grouped together into a larger region based on the distance between them)
For each cluster found in a joint signal, the cluster sheet includes the following information:
    (1) Start frame of cluster
    (2) End frame of cluster
    (3) Number of gaps the cluster contains
    (4) Duration of the cluster (in frames)
    (5) Duration of the cluster (in minutes:seconds)

'''

def import_data(dyad_file_path):
    # Extracts required attributes from .mat files for infant and parent 
        # Input: full file path for .mat file of dyad in video
        # Output: dictionary with two separate keypoint timeseries and the corresponding label (infant/parent)
        
    keypoints_mat = sio.loadmat(dyad_file_path)
    person_0 = np.array(keypoints_mat["person_0_2d"])
    person_1 = np.array(keypoints_mat["person_1_2d"])
            
    dyad_keypoints = {"infant": None, "parent": None}
    total_keypoints = [person_0, person_1]
    labels = get_subject_label(person_0, person_1)
            
    for keypoints, label in zip(total_keypoints, labels):
        dyad_keypoints[label] = keypoints
                
    return dyad_keypoints
    
def get_video_name(dyad_file_path):
    # Extracts the video name from the .mat file path
        # Input: full file path for .mat file
        # Output: string of video name 
    
    video_filename_ext = os.path.basename(dyad_file_path)
    video_name = os.path.splitext(video_filename_ext)[0]
    
    return video_name
    
def get_dyad_number(dyad_file_name):
    # Extracts the dyad number assigned in the dataset from the video name (retrieved by using get_video_name function)
        # Input: video/file name 
        # Output: int for dyad/recording number
        
    numbers_list = re.findall(r'\d+', dyad_file_name)
    
    dyad_id = numbers_list[1]
    num_str = str(dyad_id)
    
    if num_str[0] == '0':
        unpadded_dyad_id = int(str(num_str[1] + num_str[2]))
        return unpadded_dyad_id
    else:
        return dyad_id
    
def get_subject_label(keypoints_timeseries0, keypoints_timeseries1):
    # Helper function to determine which timeseries corresponds to which subject in the dyad (infant or parent labels for each input timeseries)
        # Input: two numpy arrays of joint keypoint timeseries
        # Output: list of strings with the corresponding label (ie. ["infant", "parent"] or ["parent", "infant"])
       
    # Initialize needed variables 
    counter = 0
    head_0_y = np.nan
    head_1_y = np.nan
    
    # Choose the first non-NaN keypoint in the timeseries (account for missing data at the beginning of the signal)
    while np.isnan(head_0_y):
        head_0_y = keypoints_timeseries0[15, 1, counter]
        counter += 1
    
    counter = 0
    while np.isnan(head_1_y):
        head_1_y = keypoints_timeseries1[15, 1, counter]
        counter += 1
        
    # A greater y-value in the head keypoint would correspond to the parent TM
    if head_0_y < head_1_y:
        return ["parent", "infant"]
    elif head_1_y < head_0_y:
        return ["infant", "parent"]
    else:
        print("One of the timeseries may be for a detection other than the experiment subjects")
        return None
        
def calculate_video_duration(parent_keypoint_timeseries):
    # Returns the duration of the video based on the length of the timeseries (in frames)
        # Input: list or numpy array that represents the parent keypoint timeseries (more reliable than infant keypoints as the latter may go undetected at the end and result in a shorter TS)
        # Output: integer that presents the total number of frames in the video 
    
    return parent_keypoint_timeseries.shape[2]
        
def calculation_gap_length(missing_segments_list):
    # Determines the length of the gap (ie. number of missing values) for each gap in the input list of missing segments 
        # Input: list of lists or numpy arrays (ie. segments) of missing values 
        # Output: list of integers where each elements represents the number of missing values in the corresponding segment 
        
    segment_lengths = []
    for segment in missing_segments_list:
        segment_lengths.append(len(segment.tolist()))
    
    return segment_lengths

def find_num_missing_gaps(missing_segments_list):
    # Return the number of gaps of missing data 
        # Input: list of missing data segments
        # Output: int that represents the number of missing data segments 
        
    return len(missing_segments_list)
        
def convert_length_to_time(total_video_duration_in_frames, duration_in_frames):
    # Converts the duration of a segment/video in frames to a minutes:seconds format based on the total video duration in frames    
    # assumes that every video is exactly 4 minutes long 
        # Input: int represent the duration in frames
        # Output: string of the format minutes:seconds
    
    fs = total_video_duration_in_frames/VIDEO_DURATION
    
    minutes = 0
    seconds = 0
        
    total_duration_seconds = math.ceil(duration_in_frames/fs)
    
    if (total_duration_seconds % 60) == 0:
        minutes = int(total_duration_seconds/60)
        duration = f"{minutes}:00"
    elif (total_duration_seconds % 60) < 10:
        minutes = int(math.floor(total_duration_seconds/60))
        seconds = total_duration_seconds - (60 * minutes)
        duration = f"{minutes}:0{seconds}"
    else:
        minutes = int(math.floor(total_duration_seconds/60))
        seconds = total_duration_seconds - (60 * minutes)
        duration = f"{minutes}:{seconds}"
        
    return duration

def get_gap_locations(missing_segments_list):
    # Extracts the start and end frames of each gap identified in the list of missing segments 
        # Input: list/array of missing segments 
        # Output: ints that represent the start and end frame indices of each missing gap
        
    start_frames = []
    end_frames = []
    
    for segment in missing_segments_list:
        segment = segment.tolist()
        start = segment[0]
        start_frames.append(start)
        end = segment[-1]
        end_frames.append(end)
        
    return start_frames, end_frames

def get_g2g_duration(missing_segments_list):
    # Calculates the number of frames between gaps (gap-to-gap duration) by taking the difference of the start frame in the next segment
    # and end frame of the previous segment (for n segments in missig_segments_list, n-1 durations are found)
        # Input: list/array of missing segmenets
        # Output: list/array of durations
    
    gap_durations = []
    
    # segment is taken as the current segment 
    for i in range(len(missing_segments_list) - 1):
        previous_segment = missing_segments_list[i].tolist()
        next_segment = missing_segments_list[i + 1].tolist()

        duration_start = previous_segment[-1]
        duration_end = next_segment[0]

        duration = duration_end - duration_start - 1
        gap_durations.append(duration)

    return gap_durations

def detect_gap_clusters(start_frames, end_frames, threshold, video_duration_in_frames):
    # Groups smaller gaps into clusters if the gap-to-gap distance is less than threshold
        # Input: two lists that contain the start frames and end frames of each gap, int for the maximum number of frames 
        # that can exist between two gaps for them to be considered part of the same cluster
        # Output: list of clusters (smaller gaps grouped together)
        
    clusters = []
    
    if len(start_frames) == 0:
        return clusters

    cluster_start = start_frames[0]
    cluster_end = end_frames[0]
    gaps_in_cluster = 1

    for i in range(1, len(start_frames)):

        gap_distance = start_frames[i] - cluster_end

        if gap_distance < threshold:
            # same cluster
            cluster_end = end_frames[i]
            gaps_in_cluster += 1
        else:
            # finish current cluster
            cluster_length = cluster_end - cluster_start + 1
            cluster_start_minutes = convert_length_to_time(video_duration_in_frames, cluster_start)
            cluster_end_minutes = convert_length_to_time(video_duration_in_frames, cluster_end)
            cluster_length_minutes = convert_length_to_time(video_duration_in_frames, cluster_length)
            clusters.append((cluster_start, cluster_start_minutes, cluster_end, cluster_end_minutes, gaps_in_cluster, cluster_length, cluster_length_minutes))

            # start new cluster
            cluster_start = start_frames[i]
            cluster_end = end_frames[i]
            gaps_in_cluster = 1

    # append last cluster
    cluster_length = cluster_end - cluster_start + 1
    cluster_start_minutes = convert_length_to_time(video_duration_in_frames, cluster_start)
    cluster_end_minutes = convert_length_to_time(video_duration_in_frames, cluster_end)
    cluster_length_minutes = convert_length_to_time(video_duration_in_frames, cluster_length)
    clusters.append((cluster_start, cluster_start_minutes, cluster_end, cluster_end_minutes, gaps_in_cluster, cluster_length, cluster_length_minutes))

    return clusters

def main():
    keypoints_path = "/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints"
    row_num = 0 

    infant_masterfile_data = []
    parent_masterfile_data = []
    infant_cluster_data = []
    parent_cluster_data = []
    masterfile_columns = ['Dyad Number', 'Video Name', 'Video Duration (Frames)', 'Video Duration (Minutes)', 'Joint Name', '# of Missing Data Gaps', 'Gap Duration (Frames)', 
                          'Gap Duration (Minutes)', 'Gap Start Frame', 'Gap End Frame', 'Gap Start Time (Minutes)', 'Gap End Time (Minutes)'] 
    fs = 30 # the specific FPS value for the video can be found but this would require the video path as well
    
    for file in sorted(os.listdir(keypoints_path)):

        # Import keypoints 
        file_path = os.path.join(keypoints_path, file)
        dyad_keypoints = import_data(file_path)
        infant_keypoints, parent_keypoints = dyad_keypoints["infant"], dyad_keypoints["parent"]
        
        # Get video name, dyad number, and video duration (in frames)
        video_name = get_video_name(file_path)
        dyad_number = get_dyad_number(video_name)
        video_duration_frames = calculate_video_duration(parent_keypoints)
        video_duration_in_minutes = convert_length_to_time(video_duration_frames, video_duration_frames)
        
        # Get missing gaps for infant and parent keypoints PER JOINT
        for i, joint_idx in enumerate(JOINT_INDICES):
            
            joint_name = JOINT_NAMES[i]
            infant_signal_x, infant_x_nan = replace_missing(infant_keypoints[joint_idx, 0, :])
            infant_signal_y, infant_y_nan = replace_missing(infant_keypoints[joint_idx, 1, :])
            parent_signal_x, parent_x_nan = replace_missing(parent_keypoints[joint_idx, 0, :])
            parent_signal_y, parent_y_nan = replace_missing(parent_keypoints[joint_idx, 1, :])
            
            infant_x_missing_gaps = find_missing_segments_indices(infant_signal_x)
            infant_y_missing_gaps = find_missing_segments_indices(infant_signal_y)
            parent_x_missing_gaps = find_missing_segments_indices(parent_signal_x)
            parent_y_missing_gaps = find_missing_segments_indices(parent_signal_y)
            
            # Check to ensure that x and y coordinates are both missing (usually the case with MeTRABS)
            if find_num_missing_gaps(infant_x_missing_gaps) > find_num_missing_gaps(infant_y_missing_gaps):
                infant_missing_gaps = infant_x_missing_gaps
                num_infant_missing_gaps = find_num_missing_gaps(infant_x_missing_gaps)
            else:
                infant_missing_gaps = infant_y_missing_gaps
                num_infant_missing_gaps = find_num_missing_gaps(infant_y_missing_gaps)
                
            if find_num_missing_gaps(parent_x_missing_gaps) > find_num_missing_gaps(parent_y_missing_gaps):
                parent_missing_gaps = parent_x_missing_gaps
                num_parent_missing_gaps = find_num_missing_gaps(parent_x_missing_gaps)
            else:
                parent_missing_gaps = parent_y_missing_gaps
                num_parent_missing_gaps = find_num_missing_gaps(parent_y_missing_gaps)
            
            # Calculate number of data gaps 
            infant_missing_segments_list_frames = calculation_gap_length(infant_missing_gaps)
            parent_missing_segments_list_frames = calculation_gap_length(parent_missing_gaps)
            
            # Convert the length of each gap in frames to minutes:seconds format
            infant_missing_segments_list_minutes = [convert_length_to_time(video_duration_frames, gap) for gap in infant_missing_segments_list_frames]
            parent_missing_segment_list_minutes = [convert_length_to_time(video_duration_frames, gap) for gap in parent_missing_segments_list_frames]
            
            # Extract start and end frames of each segment of missing data 
            infant_gap_start_frames, infant_gap_end_frames = get_gap_locations(infant_missing_gaps)
            parent_gap_start_frames, parent_gap_end_frames = get_gap_locations(parent_missing_gaps)
            
            # Extract start and end timestamps for each gap of missing data
            infant_gap_start_minutes = [convert_length_to_time(video_duration_frames, start_frame) for start_frame in infant_gap_start_frames]
            infant_gap_end_minutes = [convert_length_to_time(video_duration_frames, end_frame) for end_frame in infant_gap_end_frames]
            parent_gap_start_minutes = [convert_length_to_time(video_duration_frames, start_frame) for start_frame in parent_gap_start_frames]
            parent_gap_end_minutes =[convert_length_to_time(video_duration_frames, end_frame) for end_frame in parent_gap_end_frames]
            
            # Detect clusters for infant and parent signals 
            infant_clusters = detect_gap_clusters(infant_gap_start_frames, infant_gap_end_frames, CLUSTER_THRESHOLD_FRAMES, video_duration_frames)
            parent_clusters  = detect_gap_clusters(parent_gap_start_frames, parent_gap_end_frames, CLUSTER_THRESHOLD_FRAMES, video_duration_frames)
            
            # For the cluster sheets, make a row for each cluster found in each subject
            for cluster in infant_clusters:
                infant_cluster_data.append([dyad_number, video_name, video_duration_frames, video_duration_in_minutes, joint_name, cluster[0], cluster[1], cluster[2], cluster[3], cluster[4], cluster[5], cluster[6]])
                
            for cluster in parent_clusters:
                parent_cluster_data.append([dyad_number, video_name, video_duration_frames, video_duration_in_minutes, joint_name, cluster[0], cluster[1], cluster[2], cluster[3], cluster[4], cluster[5], cluster[6]])
            
            # Make a row for each gap 
            for i in range(num_infant_missing_gaps):
                 # Prepare initial data for one row (usually the same for each joint except the number of missing gaps)
                infant_row_data = [dyad_number, video_name, video_duration_frames, video_duration_in_minutes, joint_name, num_infant_missing_gaps]
                gap_info = [infant_missing_segments_list_frames[i], infant_missing_segments_list_minutes[i], infant_gap_start_frames[i], infant_gap_end_frames[i], infant_gap_start_minutes[i], infant_gap_end_minutes[i]]
                infant_row_data.extend(gap_info)
                infant_masterfile_data.append(infant_row_data)
                  
            for i in range(num_parent_missing_gaps):
                parent_row_data = [dyad_number, video_name, video_duration_frames, video_duration_in_minutes, joint_name, num_parent_missing_gaps]
                gap_info =[parent_missing_segments_list_frames[i], parent_missing_segment_list_minutes[i], parent_gap_start_frames[i], parent_gap_end_frames[i], parent_gap_start_minutes[i], parent_gap_end_minutes[i]]
                parent_row_data.extend(gap_info)
                parent_masterfile_data.append(parent_row_data)
    
    add_cluster_columns = ["Cluster Start Frame", "Cluster Start Time (Minutes)", "Cluster End Frame", "Cluster End Time (Minutes)", "# of Gaps in Cluster", "Cluster Duration (Frames)", "Cluster Duration (Minutes)"]
    cluster_columns = masterfile_columns[:-1] + add_cluster_columns
        
    # Convert both dictionaries into Dataframes 
    infant_masterfile_df = pd.DataFrame(infant_masterfile_data, columns=masterfile_columns)
    parent_masterfile_df = pd.DataFrame(parent_masterfile_data, columns=masterfile_columns)
    # infant_cluster_df = pd.DataFrame(infant_cluster_data, columns=cluster_columns)
    # parent_cluster_df = pd.DataFrame(parent_cluster_data, columns=cluster_columns)
    
    masterfile_path = '3HYPER Joint Keypoint Missing Data Gap MasterFile 5.xlsx'
    with pd.ExcelWriter(masterfile_path, engine='openpyxl') as writer:
        infant_masterfile_df.to_excel(writer, sheet_name='Gap (Infant)', index=False)
        parent_masterfile_df.to_excel(writer, sheet_name='Gap (Parent)', index=False)
        # infant_cluster_df.to_excel(writer, sheet_name='Cluster (Infant)', index=False)
        # parent_cluster_df.to_excel(writer, sheet_name='Cluster (Parent)', index=False)
        
if __name__ == '__main__':
    main()
    
        
        
