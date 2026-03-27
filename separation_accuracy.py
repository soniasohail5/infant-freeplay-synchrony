import os
import json
import numpy as np

folder_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS/Processed Data"
threshold = 40.0  # pixels allowed deviation for pelvis
pelvis_index = 1  # SMPL_24 pelvis keypoint index (0-based)
min_ids = 2  # minimum number of persons for reference frame
percentage_list = []


def load_frames_from_json(folder):
    frame_files = sorted([f for f in os.listdir(folder) if f.endswith(".json")])
    frames = []
    invalid_frames = set()  # Frames with duplicate IDs

    for idx, file in enumerate(frame_files):
        with open(os.path.join(folder, file), "r") as f:
            data = json.load(f)

        frame_dict = {}
        ids_seen = set()
        duplicate_found = False

        for person in data.get("people", []):
            person_id = person.get("person_id", None)
            if person_id is None:
                continue  # skip if no id
            if person_id in ids_seen:
                duplicate_found = True
            ids_seen.add(person_id)

            poses2d = np.array(person.get("poses2d", []))
            if poses2d.size == 0:
                continue  # skip if no 2d poses

            frame_dict[person_id] = poses2d  # shape (num_keypoints, 2)

        if duplicate_found:
            invalid_frames.add(idx)

        frames.append(frame_dict)

    return frames, invalid_frames

def pelvis_distance_2d(kp1, kp2, pelvis_idx):
    return np.linalg.norm(kp1[pelvis_idx] - kp2[pelvis_idx])

for folder in os.listdir(folder_dir)[0:49]:
    folder_path = os.path.join(folder_dir, folder)
    print(f"\nOpening folder .... {folder_path}")

    frames, invalid_frames = load_frames_from_json(folder_path)
    total_frames = len(frames)

    if total_frames == 0:
        raise ValueError("No JSON frames found or folder empty")

 
    reference_frame_idx = None
    reference_ids = None
    reference_data = None

    for idx, frame in enumerate(frames):
        if len(frame) >= min_ids:
            reference_frame_idx = idx
            reference_ids = set(frame.keys())
            reference_data = frame
            break

    if reference_frame_idx is None:
        print("No frame found with at least two persons. Skipping this folder.")
        continue

    print(f"Reference Frame: {reference_frame_idx}, IDs: {reference_ids}")

 
    correct_frames = []
    valid_frame_count = 0  # frames without duplicates for percentage calc

    for idx, frame in enumerate(frames):
        if idx in invalid_frames:  # Skip frames with duplicate IDs
            continue
        valid_frame_count += 1

        # Frame must have all reference IDs (extra persons allowed)
        if not reference_ids.issubset(frame.keys()):
            continue

        consistent = True
        for ref_id in reference_ids:
            dist = pelvis_distance_2d(frame[ref_id], reference_data[ref_id], pelvis_index)
            if dist > threshold:
                consistent = False
                break

        if consistent:
            correct_frames.append(idx)

    num_correct = len(correct_frames)
    denominator = valid_frame_count if valid_frame_count > 0 else total_frames
    percentage_correct = (num_correct / denominator) * 100
    percentage_list.append(percentage_correct)

    print(f"Number of correctly labeled frames: {num_correct}")
    print(f"Valid frames (no duplicate IDs): {valid_frame_count}")
    print(f"Frames skipped due to duplicate IDs: {len(invalid_frames)}")
    print(f"Percentage correctly labeled: {percentage_correct:.2f}%")

# Calculate average percentage across all videos
avg_percentage = (sum(percentage_list) / len(percentage_list)) if percentage_list else 0
print(f"\nAverage Percentage across all folders: {avg_percentage:.2f}%")


