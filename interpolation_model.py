import numpy as np
import scipy.io as sio
from missing_gaps_stats import load_data
from signal_plotting import find_missing_segments_indices

'''
Models the performance of thresholded linear interpolation by extracting and analyzing the following metrics:

(1) Number of gaps filled per video (achieved by plotting a stacked bar graph which shows the number of gaps filled by the threshold over the total number of gaps)
(2) Total duration of the video preserved 
(3) From (2), the percentage of the video that was interpolated over (allows us to evaluate how likely it is that the data is actually meaningful)

Provides some clues as to what dyads can be excluded from the dataset as a result of not having enough data to obtain meaningful results from
'''


