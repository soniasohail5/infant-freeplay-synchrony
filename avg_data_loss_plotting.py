import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Plots the mean data loss percentage against gap size threshold for infant and parent,
# averaged across all videos in the dataset, to identify the inflection point in the curve
# where additional threshold relaxation yields diminishing returns in data recovery.

# Reads from the aggregated summary sheets produced by data_loss.py
# Input file: interpolation_data_loss_head.xlsx

# PLOT PARAMETERS
THRESHOLD_MARKERS = [11, 13, 20]    # candidate threshold values to mark with vertical lines
INFANT_COLOR = "#2196F3"               # blue for infant
PARENT_COLOR = "#F44336"               # red for parent
SHADE_ALPHA = 0.15                     # opacity for the SD shaded band
MARKER_COLOR = "#888888"               # grey for vertical threshold marker lines
FIGURE_DPI = 150

EXCEL_PATH = "interpolation_data_loss_head.xlsx"
OUTPUT_PATH = "median_data_loss_curve_head.png"
INFLECTION_CUTOFF = 0.3            # minimum drop in data loss % per frame to consider meaningful for inflection point


def load_aggregated_summary(excel_path):
    # Loads the aggregated summary sheets for infant and parent from the data loss Excel file
        # Input: string for the path to the Excel file produced by data_loss.py
        # Output: two pandas DataFrames for infant and parent aggregated summaries

    infant_df = pd.read_excel(excel_path, sheet_name="Aggregated Summary (Infant)")
    parent_df = pd.read_excel(excel_path, sheet_name="Aggregated Summary (Parent)")

    return infant_df, parent_df


def add_threshold_markers(ax, threshold_values, y_max):
    # Adds vertical dashed lines and frame labels at each candidate threshold value
        # Input: matplotlib Axes object, list of int threshold values, float y_max
        # Output: None (modifies axes in place)

    for threshold in threshold_values:
        ax.axvline(x=threshold, color=MARKER_COLOR, linestyle="--", linewidth=0.9, alpha=0.6)
        ax.text(threshold + 0.3, y_max * 0.97, f"{threshold}f", fontsize=7.5,
                color=MARKER_COLOR, va="top")
 
def find_inflection_point(gap_thresholds, mean_curve, cutoff):
    # Finds the first threshold value where the per-frame drop in data loss falls below the cutoff
        # Input: list of int gap threshold values, 1D numpy array of mean data loss values,
        # float for the minimum drop per frame considered meaningful
        # Output: int for the threshold frame value at the inflection point, or None if not found
 
    derivatives = np.diff(mean_curve)   # rate of change between consecutive threshold values
 
    for i, derivative in enumerate(derivatives):
        if abs(derivative) < cutoff:
            return gap_thresholds[i + 1]
 
    return None
 
 
def annotate_inflection_point(ax, inflection_frame, mean_curve, gap_thresholds, color, label):
    # Adds a vertical line and annotation marking the inflection point on a data loss curve
        # Input: matplotlib Axes object, int for the inflection frame, 1D numpy array of mean data loss values,
        # list of gap threshold values, string for the curve color, string for the subject label
        # Output: None (modifies axes in place)
 
    if inflection_frame is None:
        return
 
    frame_idx = gap_thresholds.index(inflection_frame)
    y_at_inflection = mean_curve[frame_idx]
 
    ax.axvline(x=inflection_frame, color=color, linestyle="-.", linewidth=1.2, alpha=0.8)
    ax.annotate(
        f"{label} inflection\n{inflection_frame}f ({y_at_inflection:.1f}%)",
        xy=(inflection_frame, y_at_inflection),
        xytext=(inflection_frame + 1.5, y_at_inflection + 0.5),
        fontsize=7.5,
        color=color,
        arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
    )
 
 
def plot_average_data_loss(infant_df, parent_df):
    # Plots mean data loss curves with SD bands for infant and parent against gap size threshold
        # Input: two pandas DataFrames containing the aggregated summary for infant and parent
        # Output: None (displays and saves the figure)

    gap_thresholds = infant_df["gap_threshold"].tolist()
    infant_median = infant_df["median_data_loss_pct"].to_numpy()
    infant_std = infant_df["std_data_loss_pct"].to_numpy()
    parent_median = parent_df["median_data_loss_pct"].to_numpy()
    parent_std = parent_df["std_data_loss_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5), dpi=FIGURE_DPI)

    # Plot median curves
    ax.plot(gap_thresholds, infant_median, color=INFANT_COLOR, linewidth=2, label="Infant (median)")
    ax.plot(gap_thresholds, parent_median, color=PARENT_COLOR, linewidth=2, label="Parent (median)")

    y_max = max(np.max(infant_median + infant_std), np.max(parent_median + parent_std))
    add_threshold_markers(ax, THRESHOLD_MARKERS, y_max)
 
    ax.set_title("Median Data Loss % vs Gap Size Threshold — Head Keypoint", fontsize=12, fontweight="bold")
    ax.set_xlabel("Max Gap Size Threshold (frames)", fontsize=10)
    ax.set_ylabel("Median Data Loss (%)", fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_title("Median Data Loss % vs Gap Size Threshold — Head Keypoint", fontsize=12, fontweight="bold")
    ax.set_xlabel("Max Gap Size Threshold (frames)", fontsize=10)
    ax.set_ylabel("Median Data Loss (%)", fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Plot saved to {OUTPUT_PATH}")
    plt.show()


def main():
    infant_df, parent_df = load_aggregated_summary(EXCEL_PATH)
    plot_average_data_loss(infant_df, parent_df)


if __name__ == "__main__":
    main()