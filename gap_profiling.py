import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from signal_postprocessing import replace_missing
from signal_plotting import find_missing_segments_indices
from data_loss_histogram import load_gap_data

GAP_THRESHOLD_FRAMES = 12
GAP_THRESHOLD_SECONDS = 0.4
VIDEO_DURATION = 240

'''
Developing an exclusion criteria 
'''

EXCEL_PATH = "3HYPER Joint Keypoint Missing Data Gap MasterFile 5.xlsx"

def plot_gap_distribution(gap_data_df: pd.DataFrame, dyad_number: int, subject_label: str, gap_threshold_frames: int):
    
    dyad_gap_data = gap_data_df[gap_data_df["Dyad Number"] == dyad_number]
    dyad_gap_frames = dyad_gap_data["Gap Duration (Frames)"].dropna()
    
    num_bins = dyad_gap_data["Gap Duration (Frames)"].max()
    median_gap = dyad_gap_data["Gap Duration (Frames)"].median()
    mean_gap = dyad_gap_data["Gap Duration (Frames)"].mean()
    
    print(f"Median Gap Duration: {median_gap}")
    print(f"Mean Gap Duration: {mean_gap}")
    
    plt.hist(dyad_gap_frames, bins=num_bins, color='red', alpha=0.6, linewidth=0.5)
    plt.title(f"Distribution of Missing Gaps in {subject_label} of Dyad {dyad_number} (bins={num_bins}, threshold={gap_threshold_frames})", fontweight="bold")
    plt.ylabel("Number of Gaps")
    plt.xlim(1, num_bins)
    plt.axvline(x=gap_threshold_frames, color='grey', linestyle='--', linewidth=0.8, label=f'Threshold={GAP_THRESHOLD_FRAMES}')

    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.show()
    

def main():
    
    infant_df, parent_df = load_gap_data(EXCEL_PATH, "Gap (Infant)", "Gap (Parent)")
    
    # Plot gap disctribution for infant of dyad 112
    plot_gap_distribution(infant_df, 112, "Infant", GAP_THRESHOLD_FRAMES)
    # plot_gap_distribution(parent_df, 78, "Parent", GAP_THRESHOLD_FRAMES)
    
if __name__ == "__main__":
    main()
    

    

    

    
    
    

