import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plots a 2x4 figure modelling the performance of thresholded linear interpolation per video,
# used to identify dyads that may need to be excluded from the dataset.

# Layout: 2 rows (infant, parent) x 4 columns
    # Column 1: Stacked bar chart — number of gaps accepted vs rejected per dyad
    # Column 2: Horizontal bar chart — largest gap rejected per dyad (in frames)
    # Column 3: Horizontal bar chart — duration preserved after interpolation (in seconds, annotated in min:sec)
    # Column 4: Horizontal bar chart — percentage of video preserved after interpolation
    # Dyads sorted by dyad number across all plots

# Input: two pandas DataFrames produced by the interpolation model script,
# one for infant and one for parent, containing per-dyad interpolation summary statistics

# Plot parameters
GAP_THRESHOLD_FRAMES = 12               # threshold used for interpolation (for reference line annotation)
FPS = 30                                # frame rate for converting frames to seconds
VIDEO_DURATION_SECONDS = 240            # expected video duration in seconds (4 minutes)

ACCEPTED_COLOR = "#2196F3"             # blue for accepted gaps
REJECTED_COLOR = "#F44336"             # red for rejected gaps
LARGEST_GAP_COLOR = "#E65100"          # dark orange for largest rejected gap bars
PRESERVED_COLOR = "#43A047"            # green for preserved duration bars
PRESERVED_PCT_COLOR = "#1565C0"        # dark blue for preserved percentage bars
THRESHOLD_COLOR = "#333333"            # dark grey for threshold reference lines
FIGURE_DPI = 150

OUTPUT_PATH = "interpolation_model_per_video.png"


def sort_dataframe_by_dyad(df):
    # Sorts a per-dyad DataFrame by dyad number in ascending order
        # Input: pandas DataFrame with a dyad_number column
        # Output: pandas DataFrame sorted by dyad number, index reset

    return df.sort_values("dyad_number").reset_index(drop=True)


def frames_to_minsec(total_seconds):
    # Converts a duration in seconds to a minutes:seconds formatted string
        # Input: float or int for total seconds
        # Output: string in the format M:SS

    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)

    if seconds < 10:
        return f"{minutes}:0{seconds}"
    else:
        return f"{minutes}:{seconds}"


def plot_stacked_gap_bars(ax, df, subject_label):
    # Plots a stacked bar chart of accepted vs rejected gaps per dyad
        # Input: matplotlib Axes object, pandas DataFrame with per-dyad gap counts, string for subject label
        # Output: None (modifies axes in place)

    dyad_numbers = df["dyad_number"].astype(str).tolist()
    accepted = df["num_gaps_accepted"].tolist()
    rejected = df["num_gaps_rejected"].tolist()
    x = np.arange(len(dyad_numbers))

    ax.bar(x, accepted, width=0.6, color=ACCEPTED_COLOR, label="Accepted")
    ax.bar(x, rejected, width=0.6, bottom=accepted, color=REJECTED_COLOR, alpha=0.8, label="Rejected")

    ax.set_title(f"{subject_label}", fontsize=9, fontweight="bold")
    ax.set_ylabel("Number of Gaps", fontsize=8)
    ax.set_xlabel("Dyad Number", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(dyad_numbers, rotation=90, fontsize=6.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")


def plot_largest_gap_bars(ax, df, subject_label):
    # Plots a horizontal bar chart of the largest rejected gap per dyad in frames,
    # sorted from largest to smallest so the most problematic dyads appear at the top
        # Input: matplotlib Axes object, pandas DataFrame with per-dyad largest rejected gap, string for subject label
        # Output: None (modifies axes in place)

    df_sorted = df.sort_values("largest_gap_rejected", ascending=True).reset_index(drop=True)
    dyad_numbers = df_sorted["dyad_number"].astype(str).tolist()
    largest_gaps = df_sorted["largest_gap_rejected"].tolist()
    y = np.arange(len(dyad_numbers))

    ax.barh(y, largest_gaps, height=0.6, color=LARGEST_GAP_COLOR, alpha=0.85)
    ax.axvline(x=GAP_THRESHOLD_FRAMES, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2,
               label=f"Threshold = {GAP_THRESHOLD_FRAMES}f")

    for i, gap in enumerate(largest_gaps):
        if gap > GAP_THRESHOLD_FRAMES:
            ax.text(gap + 1, i, f"{gap}f", va="center", fontsize=6, color=LARGEST_GAP_COLOR)

    ax.set_title(f"{subject_label}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Largest Gap Rejected (frames)", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(dyad_numbers, fontsize=6.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="x")


def plot_preserved_duration_bars(ax, df, subject_label):
    # Plots a horizontal bar chart of the total duration preserved per dyad in seconds,
    # with each bar annotated in minutes:seconds format
        # Input: matplotlib Axes object, pandas DataFrame with per-dyad preserved duration in frames,
        # string for subject label
        # Output: None (modifies axes in place)

    dyad_numbers = df["dyad_number"].astype(str).tolist()
    preserved_seconds = (df["preserved_duration_frames"] / FPS).tolist()
    y = np.arange(len(dyad_numbers))

    ax.barh(y, preserved_seconds, height=0.6, color=PRESERVED_COLOR, alpha=0.85)

    # Reference line at expected full video duration
    ax.axvline(x=VIDEO_DURATION_SECONDS, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2,
               label=f"Full duration = {frames_to_minsec(VIDEO_DURATION_SECONDS)}")

    # Annotate each bar with min:sec format
    for i, seconds in enumerate(preserved_seconds):
        ax.text(seconds + 1, i, frames_to_minsec(seconds), va="center", fontsize=6, color=PRESERVED_COLOR)

    ax.set_title(f"{subject_label}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Duration Preserved (seconds)", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(dyad_numbers, fontsize=6.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="x")


def plot_interpolated_percentage_of_preserved_bars(ax, df, subject_label):
    # Plots a horizontal bar chart of the percentage of preserved data that was interpolated per dyad,
    # to identify videos where a large proportion of usable data is reconstructed rather than observed
        # Input: matplotlib Axes object, pandas DataFrame with per-dyad preserved and interpolated frame counts,
        # string for subject label
        # Output: None (modifies axes in place)

    dyad_numbers = df["dyad_number"].astype(str).tolist()
    interpolated_pct_of_preserved = (df["interpolated_frames"] / df["preserved_duration_frames"] * 100).round(2).tolist()
    y = np.arange(len(dyad_numbers))

    ax.barh(y, interpolated_pct_of_preserved, height=0.6, color=PRESERVED_PCT_COLOR, alpha=0.85)

    for i, pct in enumerate(interpolated_pct_of_preserved):
        if pct > 0:
            ax.text(pct + 0.3, i, f"{pct:.1f}%", va="center", fontsize=6, color=PRESERVED_PCT_COLOR)

    ax.set_title(f"{subject_label}", fontsize=9, fontweight="bold")
    ax.set_xlabel("Interpolated Frames / Preserved Frames (%)", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(dyad_numbers, fontsize=6.5)
    ax.grid(True, alpha=0.3, axis="x")


def plot_interpolation_model(infant_df, parent_df):
    # Produces a 2x4 figure showing interpolation model performance per dyad for infant and parent
        # Input: two pandas DataFrames containing per-dyad interpolation summary for infant and parent
        # Output: None (displays and saves the figure)

    infant_df = sort_dataframe_by_dyad(infant_df)
    parent_df = sort_dataframe_by_dyad(parent_df)

    fig, axes = plt.subplots(2, 4, figsize=(26, 14), dpi=FIGURE_DPI)
    fig.suptitle(
        f"Interpolation Model Performance per Video — Head Keypoint  (Threshold = {GAP_THRESHOLD_FRAMES}f / {GAP_THRESHOLD_FRAMES/FPS:.2f}s)",
        fontsize=13, fontweight="bold"
    )

    # Row 0: infant
    plot_stacked_gap_bars(axes[0, 0], infant_df, "Gaps Accepted vs Rejected")
    plot_largest_gap_bars(axes[0, 1], infant_df, "Largest Gap Rejected (frames)")
    plot_preserved_duration_bars(axes[0, 2], infant_df, "Duration Preserved")
    plot_interpolated_percentage_of_preserved_bars(axes[0, 3], infant_df, "% of Preserved Data That Was Interpolated")

    # Row 1: parent
    plot_stacked_gap_bars(axes[1, 0], parent_df, "Gaps Accepted vs Rejected")
    plot_largest_gap_bars(axes[1, 1], parent_df, "Largest Gap Rejected (frames)")
    plot_preserved_duration_bars(axes[1, 2], parent_df, "Duration Preserved")
    plot_interpolated_percentage_of_preserved_bars(axes[1, 3], parent_df, "% of Preserved Data That Was Interpolated")

    # Row labels
    axes[0, 0].annotate("Infant", xy=(-0.22, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center", ha="center",
                         color=ACCEPTED_COLOR)
    axes[1, 0].annotate("Parent", xy=(-0.22, 0.5), xycoords="axes fraction",
                         fontsize=11, fontweight="bold", rotation=90, va="center", ha="center",
                         color=REJECTED_COLOR)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(f"Plot saved to {OUTPUT_PATH}")
    plt.show()


def main():
    # Load per-dyad interpolation summary DataFrames produced by the interpolation model script
    infant_df = pd.read_excel("interpolation_model_summary.xlsx", sheet_name="Infant")
    parent_df = pd.read_excel("interpolation_model_summary.xlsx", sheet_name="Parent")

    plot_interpolation_model(infant_df, parent_df)


if __name__ == "__main__":
    main()
