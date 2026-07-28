import os 
import json

'''
General functions to retrieve the video name and .mat file given the dyad number from the JSON keypoint-video file pairs
'''

def get_video_file_path(dyad_number: int, folder_path: str) -> str:
    
    if os.path.isdir(folder_path) == False:
        raise FileNotFoundError("Directory is not found. Check if the folder path corresponds with the path of the connected external drive and ensure it is mounted in WSL Ubuntu instance.")
    
    with open("video_dict.json", "r") as f:
        data_table = json.load(f)
        
    dyad_number_str = str(dyad_number)
    
    if dyad_number_str not in data_table:
        raise KeyError("Dyad number not found.")
    
    if "video_name" not in data_table[dyad_number_str]:
        raise KeyError(f"Video for Dyad {dyad_number} was not found.")
    
    dyad_video_file = data_table[dyad_number_str]["video_name"]
    dyad_video_file_path = os.path.join(folder_path, dyad_video_file) + ".mp4"
    
    if os.path.isfile(dyad_video_file_path) == False:
        raise FileNotFoundError(f"{dyad_video_file} was not found in directory.")
    
    return dyad_video_file_path
    
def get_mat_file_path(dyad_number: int, folder_path: str, ) -> str:
    
    keypoint_type = str(input("Enter keypoint type (2D or 3D): "))
    
    if os.path.isdir(folder_path) == False:
        raise FileNotFoundError("Directory is not found. Check if the folder path exists")
    
    with open("video_dict.json", "r") as f:
        data_table = json.load(f)
        
    dyad_number_str = str(dyad_number)
    
    if dyad_number_str not in data_table:
        raise KeyError("Dyad number not found")
    
    if keypoint_type != "3D" or keypoint_type != "3d":
        if keypoint_type != "2D" or keypoint_type != "2d":
            raise AttributeError("Keypoint type does not exist. Typo?")
        else:
            dyad_mat_file = data_table[dyad_number_str][keypoint_type]
            
    if "2D Keypoints" not in data_table[dyad_number_str] and "3D Keypoints" not in data_table[dyad_number_str]:
        raise KeyError(f"Keypoints for Dyad {dyad_number} was not found.")
            
    dyad_mat_file_path = os.path.join(folder_path, dyad_mat_file) + ".mat"
    
    if os.path.isfile(dyad_mat_file_path) == False:
        raise FileNotFoundError(f" .mat file {dyad_mat_file} was not found in directory.")
    
    return dyad_mat_file_path
    
        
    
    
        
