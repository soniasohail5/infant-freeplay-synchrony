import os
import numpy as np
import pandas as pd
from missing_gaps_stats import import_data
from signal_postprocessing import replace_missing
from signal_plotting import find_missing_segments_indices

# Calculates data loss and recovery percentages for the head keypoint based only on gap size threshold,
# averaged across all videos in the dataset.

# Data loss %  = percentage of total frames still missing after applying the gap size threshold
# Recovery %   = percentage of originally missing frames that were recovered by interpolation

# Array shape contract from MeTRABS: (joints, 2, frames) — transposed to (frames,) for the head signal

HEAD_JOINT_INDEX = 15                           # index of the head keypoint in the MeTRABS skeleton
GAP_THRESHOLDS = np.arange(5, 31).tolist()      # gap size thresholds to sweep (1 to 15 frames)
FOLDER_PATH = "/mnt/c/3HYPER FREEPLAY DV METRABS/MATLAB Keypoints 2/2D Keypoints/"

def get_file_path(folder_path, dyad_number):
    # Constructs the full file path for a given dyad number with zero-padded formatting
        # Input: string for folder path, int for dyad number
        # Output: string for the full file path

    if dyad_number >= 100:
        return folder_path + "3HYPER." + str(dyad_number) + " FREEPLAY DV EXTRACTED 2D Keypoints.mat"
    elif dyad_number >= 10:
        return folder_path + "3HYPER.0" + str(dyad_number) + " FREEPLAY DV EXTRACTED 2D Keypoints.mat"
    else:
        return folder_path + "3HYPER.00" + str(dyad_number) + " FREEPLAY DV EXTRACTED 2D Keypoints.mat"


def find_gap_indices(missing_signal):
    # Finds all contiguous runs of missing (True) values in a 1D boolean signal
        # Input: 1D boolean numpy array where True indicates a missing frame
        # Output: list of tuples (start_frame, end_frame) for each gap, inclusive

    gaps = []
    n = len(missing_signal)
    i = 0

    while i < n:
        if missing_signal[i]:
            j = i
            while j < n and missing_signal[j]:
                j += 1
            gaps.append((i, j - 1))
            i = j
        else:
            i += 1

    return gaps


def calculate_data_loss(head_signal, gap_threshold):
    # Calculates data loss and recovery percentages for a 1D head keypoint signal under a given gap size threshold
        # Input: 1D numpy array for the head signal (x or y coordinate), int for the gap size threshold
        # Output: tuple of (n_frames, originally_missing_frames, interpolated_frames, rejected_frames, data_loss_pct, recovery_pct)

    n_frames = head_signal.shape[0]
    missing_signal = np.isnan(head_signal)
    gaps = find_gap_indices(missing_signal)

    originally_missing_frames = int(missing_signal.sum())
    interpolated_frames = 0
    rejected_frames = 0

    for gap_start, gap_end in gaps:
        gap_length = gap_end - gap_start + 1
        has_left_anchor = gap_start >= 1 and not missing_signal[gap_start - 1]
        has_right_anchor = gap_end + 1 < n_frames and not missing_signal[gap_end + 1]
        if gap_length <= gap_threshold and has_left_anchor and has_right_anchor:
            interpolated_frames += gap_length
        else:
            rejected_frames += gap_length

    data_loss_pct = round(100 * (rejected_frames / n_frames), 4)

    if originally_missing_frames == 0:
        recovery_pct = 100
    else:
        recovery_pct = round(100 * (interpolated_frames / originally_missing_frames), 4)

    return n_frames, originally_missing_frames, interpolated_frames, rejected_frames, data_loss_pct, recovery_pct


def sweep_gap_thresholds_for_dyad(infant_head_signal, parent_head_signal, gap_thresholds):
    # Sweeps all gap size thresholds for one dyad and returns data loss statistics for infant and parent
        # Input: two 1D numpy arrays for infant and parent head signals, list of int gap threshold values
        # Output: two lists of dicts — one per threshold for infant, one per threshold for parent

    infant_results = []
    parent_results = []

    for gap_threshold in gap_thresholds:
        infant_n_frames, infant_originally_missing, infant_interpolated, infant_rejected, infant_data_loss_pct, infant_recovery_pct = calculate_data_loss(infant_head_signal, gap_threshold)
        parent_n_frames, parent_originally_missing, parent_interpolated, parent_rejected, parent_data_loss_pct, parent_recovery_pct = calculate_data_loss(parent_head_signal, gap_threshold)

        infant_results.append({
            "gap_threshold": gap_threshold,
            "total_frames": infant_n_frames,
            "total_missing_frames": infant_originally_missing,
            "interpolated_frames": infant_interpolated,
            "dropped_frames": infant_rejected,
            "data_loss_pct": infant_data_loss_pct,
            "recovery_pct": infant_recovery_pct,
        })
        parent_results.append({
            "gap_threshold": gap_threshold,
            "total_frames": parent_n_frames,
            "total_missing_frames": parent_originally_missing,
            "interpolated_frames": parent_interpolated,
            "dropped_frames": parent_rejected,
            "data_loss_pct": parent_data_loss_pct,
            "recovery_pct": parent_recovery_pct,
        })

    return infant_results, parent_results


def average_curves_across_dyads(all_dyad_curves):
    # Computes the mean, standard deviation, min, max, and range of a metric across all dyads at each threshold value
        # Input: list of lists where each inner list contains per-threshold values for one dyad
        # Output: five 1D numpy arrays for mean, std, min, max, and range across dyads at each threshold

    curves_array = np.array(all_dyad_curves)
    mean_curve = np.mean(curves_array, axis=0)
    median_curve = np.median(curves_array, axis=0)
    std_curve = np.std(curves_array, axis=0)
    min_curve = np.min(curves_array, axis=0)
    max_curve = np.max(curves_array, axis=0)
    range_curve = max_curve - min_curve

    return mean_curve, median_curve, std_curve, min_curve, max_curve, range_curve


def save_results_to_excel(results_df, aggregated_df, output_path):
    # Saves per-dyad and aggregated threshold sweep results to an Excel file with three sheets
        # Input: pandas DataFrame with per-dyad results, pandas DataFrame with aggregated summary,
        # string for the output file path
        # Output: None (saves file to disk)

    infant_df = results_df[results_df["subject"] == "Infant"].drop(columns="subject")
    parent_df = results_df[results_df["subject"] == "Parent"].drop(columns="subject")
    infant_aggregated_df = aggregated_df[aggregated_df["subject"] == "Infant"].drop(columns="subject")
    parent_aggregated_df = aggregated_df[aggregated_df["subject"] == "Parent"].drop(columns="subject")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        infant_df.to_excel(writer, sheet_name="Per-Dyad (Infant)", index=False)
        parent_df.to_excel(writer, sheet_name="Per-Dyad (Parent)", index=False)
        infant_aggregated_df.to_excel(writer, sheet_name="Aggregated Summary (Infant)", index=False)
        parent_aggregated_df.to_excel(writer, sheet_name="Aggregated Summary (Parent)", index=False)

    print(f"Results saved to {output_path}")


def main():
    all_infant_data_loss = []
    all_infant_recovery = []
    all_parent_data_loss = []
    all_parent_recovery = []
    all_records = []

    for file in sorted(os.listdir(FOLDER_PATH)):
        if not file.endswith(".mat"):
            continue

        file_path = os.path.join(FOLDER_PATH, file)

        dyad_keypoint_data = import_data(file_path)
        infant_keypoint_data, _ = replace_missing(dyad_keypoint_data["infant"])
        parent_keypoint_data, _ = replace_missing(dyad_keypoint_data["parent"])

        # Extract head keypoint x-coordinate signal for infant and parent
        infant_head_signal = infant_keypoint_data[HEAD_JOINT_INDEX, 0, :]
        parent_head_signal = parent_keypoint_data[HEAD_JOINT_INDEX, 0, :]

        infant_results, parent_results = sweep_gap_thresholds_for_dyad(infant_head_signal, parent_head_signal, GAP_THRESHOLDS)

        all_infant_data_loss.append([r["data_loss_pct"] for r in infant_results])
        all_infant_recovery.append([r["recovery_pct"] for r in infant_results])
        all_parent_data_loss.append([r["data_loss_pct"] for r in parent_results])
        all_parent_recovery.append([r["recovery_pct"] for r in parent_results])

        # Store per-dyad per-threshold records for Excel output
        for result in infant_results:
            all_records.append({"file": file, "subject": "Infant", **result})
        for result in parent_results:
            all_records.append({"file": file, "subject": "Parent", **result})

        print(f"Processed: {file}")

    # Average curves across all dyads
    infant_mean_data_loss, infant_median_data_loss, infant_std_data_loss, infant_min_data_loss, infant_max_data_loss, infant_range_data_loss = average_curves_across_dyads(all_infant_data_loss)
    infant_mean_recovery, infant_median_recovery, infant_std_recovery, infant_min_recovery, infant_max_recovery, infant_range_recovery = average_curves_across_dyads(all_infant_recovery)
    parent_mean_data_loss, parent_median_data_loss, parent_std_data_loss, parent_min_data_loss, parent_max_data_loss, parent_range_data_loss = average_curves_across_dyads(all_parent_data_loss)
    parent_mean_recovery, parent_median_recovery, parent_std_recovery, parent_min_recovery, parent_max_recovery, parent_range_recovery = average_curves_across_dyads(all_parent_recovery)

    # Print summary at each threshold
    print(f"\n{'Threshold':>10} {'Infant Loss%':>14} {'Infant Recov%':>15} {'Parent Loss%':>14} {'Parent Recov%':>15}")
    print("-" * 72)
    for i, gap_threshold in enumerate(GAP_THRESHOLDS):
        print(
            f"{gap_threshold:>10} "
            f"{infant_mean_data_loss[i]:>12.2f}% "
            f"{infant_mean_recovery[i]:>13.2f}% "
            f"{parent_mean_data_loss[i]:>12.2f}% "
            f"{parent_mean_recovery[i]:>13.2f}%"
        )

    # Build aggregated summary DataFrame across dyads
    aggregated_records = []
    for i, gap_threshold in enumerate(GAP_THRESHOLDS):
        aggregated_records.append({
            "subject": "Infant",
            "gap_threshold": gap_threshold,
            "mean_data_loss_pct": round(float(infant_mean_data_loss[i]), 4),
            "median_data_loss_pct": round(float(infant_median_data_loss[i]), 4),
            "std_data_loss_pct": round(float(infant_std_data_loss[i]), 4),
            "min_data_loss_pct": round(float(infant_min_data_loss[i]), 4),
            "max_data_loss_pct": round(float(infant_max_data_loss[i]), 4),
            "range_data_loss_pct": round(float(infant_range_data_loss[i]), 4),
            "mean_recovery_pct": round(float(infant_mean_recovery[i]), 4),
            "median_recovery_pct": round(float(infant_median_recovery[i]), 4),
            "std_recovery_pct": round(float(infant_std_recovery[i]), 4),
            "min_recovery_pct": round(float(infant_min_recovery[i]), 4),
            "max_recovery_pct": round(float(infant_max_recovery[i]), 4),
            "range_recovery_pct": round(float(infant_range_recovery[i]), 4),
        })
        aggregated_records.append({
            "subject": "Parent",
            "gap_threshold": gap_threshold,
            "mean_data_loss_pct": round(float(parent_mean_data_loss[i]), 4),
            "median_data_loss_pct": round(float(parent_median_data_loss[i]), 4),
            "std_data_loss_pct": round(float(parent_std_data_loss[i]), 4),
            "min_data_loss_pct": round(float(parent_min_data_loss[i]), 4),
            "max_data_loss_pct": round(float(parent_max_data_loss[i]), 4),
            "range_data_loss_pct": round(float(parent_range_data_loss[i]), 4),
            "mean_recovery_pct": round(float(parent_mean_recovery[i]), 4),
            "median_recovery_pct": round(float(parent_median_recovery[i]), 4),
            "std_recovery_pct": round(float(parent_std_recovery[i]), 4),
            "min_recovery_pct": round(float(parent_min_recovery[i]), 4),
            "max_recovery_pct": round(float(parent_max_recovery[i]), 4),
            "range_recovery_pct": round(float(parent_range_recovery[i]), 4),
        })

    aggregated_df = pd.DataFrame(aggregated_records)

    # Save per-dyad and aggregated results to Excel
    results_df = pd.DataFrame(all_records)
    results_df = results_df[["file", "subject", "gap_threshold", "total_frames", "total_missing_frames",
                              "interpolated_frames", "dropped_frames", "data_loss_pct", "recovery_pct"]]
    save_results_to_excel(results_df, aggregated_df, "interpolation_data_loss_head.xlsx")


if __name__ == "__main__":
    main()