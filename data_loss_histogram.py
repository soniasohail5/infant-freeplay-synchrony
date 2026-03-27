import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Plots a histogram of all gap durations found across all videos for infant and parent,
# to illustrate the distribution of missing data gaps and contextualise the threshold decision.

# Reads from the gap masterfile Excel sheets produced by missing_gaps_stats.py
# Input file: excel sheet with two sheets — one for infant gaps, one for parent gaps

# PLOT PARAMETERS
GAP_DURATION_COLUMN = "Gap Duration (Frames)"      # column name for gap duration in frames
DYAD_NUMBER_COLUMN = "Dyad Number"                 # column name for dyad number
PROPOSED_THRESHOLD = 10                            # proposed interpolation threshold to mark on the plot
X_AXIS_MAX = 60                               # maximum gap duration shown on x-axis (frames)
BIN_WIDTH = 1                                      # width of each histogram bin in frames

INFANT_COLOR = "#2196F3"                           # blue for infant
PARENT_COLOR = "#F44336"                           # red for parent
THRESHOLD_COLOR = "#333333"                        # dark grey for threshold marker
FIGURE_DPI = 150

EXCEL_PATH = "3HYPER Joint Keypoint Missing Data Gap MasterFile 5.xlsx"
INFANT_SHEET = "Gap (Infant)"
PARENT_SHEET = "Gap (Parent)"
OUTPUT_PATH = "gap_duration_histogram.png"


def load_gap_data(excel_path, infant_sheet, parent_sheet):
    # Loads the infant and parent gap data from the masterfile Excel sheets
        # Input: string for the Excel file path, string for the infant sheet name, string for the parent sheet name
        # Output: two pandas DataFrames for infant and parent gap data

    infant_df = pd.read_excel(excel_path, sheet_name=infant_sheet)
    parent_df = pd.read_excel(excel_path, sheet_name=parent_sheet)

    return infant_df, parent_df


def calculate_capture_percentage(gap_durations, threshold):
    # Calculates the percentage of gaps that fall within the proposed threshold
        # Input: 1D array or list of gap durations in frames, int for the threshold
        # Output: float representing the percentage of gaps at or below the threshold

    total_gaps = len(gap_durations)
    if total_gaps == 0:
        return 0.0

    captured_gaps = int((gap_durations <= threshold).sum())
    return round(100.0 * captured_gaps / total_gaps, 1)


def plot_gap_histogram(infant_df, parent_df):
    # Plots overlaid histograms of gap durations for infant and parent with threshold annotation
        # Input: two pandas DataFrames containing the infant and parent gap masterfile data
        # Output: None (displays and saves the figure)

    # Extract gap durations and clip to x-axis max for display
    infant_gaps = infant_df[GAP_DURATION_COLUMN].dropna()
    parent_gaps = parent_df[GAP_DURATION_COLUMN].dropna()
    infant_gaps_clipped = infant_gaps[infant_gaps <= X_AXIS_MAX]
    parent_gaps_clipped = parent_gaps[parent_gaps <= X_AXIS_MAX]

    # Calculate capture percentages at proposed threshold (using full unclipped data)
    infant_capture_pct = calculate_capture_percentage(infant_gaps, PROPOSED_THRESHOLD)
    parent_capture_pct = calculate_capture_percentage(parent_gaps, PROPOSED_THRESHOLD)

    # Calculate total gap counts
    n_infant_total = len(infant_gaps)
    n_parent_total = len(parent_gaps)
    n_infant_clipped = len(infant_gaps) - len(infant_gaps_clipped)
    n_parent_clipped = len(parent_gaps) - len(parent_gaps_clipped)

    bins = np.arange(1, X_AXIS_MAX + BIN_WIDTH + 1, BIN_WIDTH) - 0.5

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), dpi=FIGURE_DPI, sharex=True)
    fig.suptitle("Distribution of Missing Data Gap Durations — Head Keypoint", fontsize=13, fontweight="bold")

    # Infant subplot
    axes[0].hist(infant_gaps_clipped, bins=bins, color=INFANT_COLOR, alpha=0.8, edgecolor="white", linewidth=0.4)
    axes[0].axvline(x=PROPOSED_THRESHOLD, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.5,
                    label=f"Proposed threshold = {PROPOSED_THRESHOLD}f ({infant_capture_pct}% of gaps captured)")
    axes[0].set_title(f"Infant  (n = {n_infant_total} total gaps)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Number of Gaps", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3, axis="y")

    if n_infant_clipped > 0:
        axes[0].text(0.98, 0.95, f"{n_infant_clipped} gaps > {X_AXIS_MAX}f not shown",
                     transform=axes[0].transAxes, fontsize=7.5, color="grey",
                     ha="right", va="top")

    # Parent subplot
    axes[1].hist(parent_gaps_clipped, bins=bins, color=PARENT_COLOR, alpha=0.8, edgecolor="white", linewidth=0.4)
    axes[1].axvline(x=PROPOSED_THRESHOLD, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.5,
                    label=f"Proposed threshold = {PROPOSED_THRESHOLD}f ({parent_capture_pct}% of gaps captured)")
    axes[1].set_title(f"Parent  (n = {n_parent_total} total gaps)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Number of Gaps", fontsize=9)
    axes[1].set_xlabel("Gap Duration (frames)", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3, axis="y")

    if n_parent_clipped > 0:
        axes[1].text(0.98, 0.95, f"{n_parent_clipped} gaps > {X_AXIS_MAX}f not shown",
                     transform=axes[1].transAxes, fontsize=7.5, color="grey",
                     ha="right", va="top")

    axes[1].xaxis.set_major_locator(ticker.MultipleLocator(5))

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Infant  — total gaps: {n_infant_total}, captured at {PROPOSED_THRESHOLD}f: {infant_capture_pct}%")
    print(f"Parent  — total gaps: {n_parent_total}, captured at {PROPOSED_THRESHOLD}f: {parent_capture_pct}%")
    print(f"Plot saved to {OUTPUT_PATH}")
    plt.show()


def main():
    infant_df, parent_df = load_gap_data(EXCEL_PATH, INFANT_SHEET, PARENT_SHEET)
    plot_gap_histogram(infant_df, parent_df)


if __name__ == "__main__":
    main()