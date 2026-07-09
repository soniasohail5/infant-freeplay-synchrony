import os
import copy
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

# Batch processing for tracking and labeling using frame differencing and the Hungarian assignment algorithm
# Hungarian is only used for frames with 3+ detections

# Helper Functions
def get_confidence_score(person_boxes):
    confidence = person_boxes[4]
    
    return confidence

def calculate_bounding_box_area(person_boxes):
    width = person_boxes[2]
    height = person_boxes[3]
    
    area = width * height
    
    return area

def calculate_person_height(joint_keypoints):
    head_kp = joint_keypoints[15, :] 
    pelvis_kp = joint_keypoints[0, :]
    
    person_height = abs(np.linalg.norm(head_kp - pelvis_kp))
    
    return person_height

# Cost function
def calculate_cost_function(unknown_person, ref_person):
    # cost will be calculated by weighted sum of the bounding box area and the person's height
    # weight for the bounding box area is dependent on the detection confidence score
    previous_bounding_box_area= calculate_bounding_box_area(ref_person["boxes"])
    bounding_box_area = calculate_bounding_box_area(unknown_person["boxes"])
    
    confidence  = get_confidence_score(unknown_person["boxes"])
    
    previous_person_height = calculate_person_height(np.array(ref_person["poses3d"]))
    person_height = calculate_person_height(np.array(unknown_person["poses3d"]))
    
    weights = [1.0, 0.6, 2.0] # Initial weights (will be adjusted as needed)
    
    # Feature Normalization
    bounding_box_area_norm = abs(bounding_box_area - previous_bounding_box_area)/previous_bounding_box_area
    person_height_norm = abs(person_height - previous_person_height)/previous_person_height
    confidence_penalty = 1 - confidence
    
    features_norm = [bounding_box_area_norm, person_height_norm, confidence_penalty]
    
    # Use dot product to calculate cost with features + confidence penalty
    cost = np.dot(weights, features_norm)
    
    return cost

# Labeling function for 1-2 detection frames (method is similar to previous)
def two_detection_labeling(current_dyad_info, previous_person_0, previous_person_1):
    people = [previous_person_0, previous_person_1]
    frame_distances = []

    for person in current_dyad_info["people"]:
        for unknown_person in people:
            current_frame = np.array(person["poses3d"])
            try:
                previous_frame = np.array(unknown_person["poses3d"])
            except IndexError:
                previous_frame = np.array(unknown_person)
                
            try:
                previous_frame = previous_frame.reshape(24, 3)
                current_frame = current_frame.reshape(24, 3)
                frame_distance = np.linalg.norm(current_frame[0, :] - previous_frame[0, :]) # use pelvis keypoint to track (most stable)
                frame_distances.append(frame_distance)

            except ValueError:
                frame_distances.append(np.nan)
            
        # For the closest person, check if the IDs are the same
        # If they are, add the joint keypoints from the person to the appropriate list (person_0, person_1 or person_2)
        # If they are not, change the ID to the correct one 
        # then add the joint keypoints to the list
        identified_person = frame_distances.index(min(frame_distances))
            
        if person["person_id"] != identified_person:
            person["person_id"] = identified_person
        
        frame_distances.clear()
        
    labeled_dyad_info = copy.deepcopy(current_dyad_info)
    
    return labeled_dyad_info

# Labeling function for 3 detection frames (uses Hungarian assignment algorithm)
def three_detection_labeling(current_dyad_info, previous_dyad_info): 
    labeled_dyad_info = {}
    
    tracked_ids = [person["person_id"] for person in previous_dyad_info["people"]]
    
    # Previous dyad_info dictionary is used as reference for the cost function 
    # Use nested for loop to build the cost matrix
    
    num_detections = len(current_dyad_info["people"])
    num_people = len(previous_dyad_info["people"])
    
    # Build the cost matrix 
    # Can also be applied for 3 detections in current, 2 in the previous (which is usually the case)
    cost_matrix = np.zeros((num_people, num_detections))
    
    for i, person in enumerate(previous_dyad_info["people"]):
        for j, detection in enumerate(current_dyad_info["people"]):
            
            cost_matrix[i, j] = calculate_cost_function(detection, person)
   
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Use row indices as correct IDs
    for r, c in zip(row_ind, col_ind):
        current_dyad_info["people"][c]["person_id"] = tracked_ids[r]
    
    # Handle unassigned detections (when det_prev < det_current)
    assigned_cols = set(col_ind)
    for j, det in enumerate(current_dyad_info["people"]):
        if j not in assigned_cols:
            det["person_id"] = 2  
            
    labeled_dyad_info = copy.deepcopy(current_dyad_info)
    
    return labeled_dyad_info

def make_ids_consistent(dyad_info):
    for person in dyad_info["people"]:
        if type(person["person_id"]) is list:
            person["person_id"] = person["person_id"][0] # changes IDs from lists to ints for easier manipulation and consistency
                
    return dyad_info

# Directory
folder_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS/labelling 772026"
new_folder_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS/Processed Data 2"

for folder in sorted(os.listdir(folder_dir)):
   # Initialize lists for infant and parent (2 detection labeling)
    # Initialize list of frames for 3 detection labeling (collection of dyad_info)
    person_0, person_1 = [], []
    dyad_info_tracking = []

    print("Opening folder: " + folder)
    folder_path = os.path.join(folder_dir, folder)
    # print(folder_path)
    
    # Number of frames for each subject
    frame_count = len(os.listdir(folder_path))
    
    dyad_info = {"people": []}
    ref_frame_count = -1
    
    # Find the first non-empty frame which will serve as the reference frame 
    # for the first 10 iterations
    while dyad_info["people"] == [] or len(dyad_info["people"]) < 2:
        ref_frame_count += 1
        ref_frame_path = os.path.join(folder_path, sorted(os.listdir(folder_path))[ref_frame_count])
        if ref_frame_path.endswith(".json"):
            ref_frame = open(ref_frame_path, 'r')
            print(ref_frame_path)
            dyad_info = json.load(ref_frame)
            ref_frame.close()
        
    # Before adding ref frame into tracking list, make all the IDs adhere to a specific format (use integers not lists)
    dyad_info = make_ids_consistent(dyad_info)
    
    # Store the entire person's dictionary into person_0, person_1
    for person in dyad_info["people"]:
        if  person["person_id"] == 0:
            person_0.append(person)
            
        elif person["person_id"] == 1:
            person_1.append(person)
            
    # Store empty reference keypoints in case either parent or infant is missing in the first/reference frame
    if len(person_0) == 0:
        person_0.append(np.zeros((24, 3)))
    if len(person_1) == 0:
        person_1.append(np.full((24, 3), 600))
        
    # Add reference frame to the dyad_info_tracking list 
   # In case if 3 detections are found in the first frame 
    dyad_info_tracking.append(dyad_info)
     
    # Frame-by-frame labeling and tracking
    for frame in sorted(os.listdir(folder_path)[ref_frame_count+1:]):
        if frame.endswith(".json"):
            open_file = open(os.path.join(folder_path, frame), 'r')
            print("Opening .... " + frame)
            dyad_info = json.load(open_file)
            open_file.close()
        else:
            continue
        
        # Standardize ID format in current frame before applying labelling algorithm
        num_detections = len(dyad_info["people"])
        dyad_info = make_ids_consistent(dyad_info)
        
        # Runs either labeling method based on the number of detections
        # 0, 1, or 2 -> 2-detection labeling method
        # 3+ -> 3-detection labeling method
        
        if num_detections <= 2:
            new_dyad_info = two_detection_labeling(dyad_info, person_0[-1], person_1[-1])
        else:
            new_dyad_info = three_detection_labeling(dyad_info, dyad_info_tracking[-1])
        
        # Adding newly labelled keypoints into person_0, person_1 to become the reference for the next frame
        for labeled_person in new_dyad_info["people"]:
            if labeled_person["person_id"] == 0:
                person_0.append(labeled_person)
            elif labeled_person["person_id"] == 1:
                person_1.append(labeled_person)
        
        # Append newly labeled frame as reference for the next frame (3-detection labeling)        
        dyad_info_tracking.append(new_dyad_info)
        
        # Keep up to the previous 10 frames in each list, when the list reaches 10 items, 
        # discard all except the last frame (the previous frame)      
        if len(person_0) > 10:
            person_0.pop(0)
        
        if len(person_1) > 10:
            person_1.pop(0)
            
        if len(dyad_info_tracking) > 10:
            dyad_info_tracking.pop(0)
          
        # Save the separated IDs in a new folder for processed data
        new_file_name = os.path.join(new_folder_dir, folder, frame.replace(".json", ".json"))
        os.makedirs(os.path.dirname(new_file_name), exist_ok=True)

        # Write the modified data into the new file
        with open(new_file_name, 'w') as new_file:
            json.dump(new_dyad_info, new_file)

            new_file.close()
            dyad_info.clear()

    print("All frames have been processed.")

print("All dyads have been processed.")

            
        
        
            