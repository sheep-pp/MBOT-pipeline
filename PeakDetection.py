#when out of bound puts it at the end of the file

import numpy as np
from scipy.signal import find_peaks

def f_find_peaks(smoothed_force, gradient_force, main_threshold=0.03, sub_threshold=0.015, distance=25):
    peaks = []  
    start_rises = [] 
    end_offsets = []  
    sub_peaks = [] 

    i = 0
    while i < len(smoothed_force):
        if smoothed_force[i] > main_threshold and gradient_force[i] >= 0:
            # find start of a peak 
            highest_peak = i
            while i < len(smoothed_force) and (smoothed_force[i] > main_threshold or gradient_force[i] >= 0):
                # Find the highest point in this peak region
                if smoothed_force[i] > smoothed_force[highest_peak]:
                    highest_peak = i
                i += 1

            # start from the highest peak, move backwards to find where the rise started
            start_rise = highest_peak
            while start_rise > 0 and (smoothed_force[start_rise] >= main_threshold or gradient_force[start_rise] >= 0):
                start_rise -= 1
            start_rise += 1  # move forward one step as the loop stops at the point just below the main_threshold or negative gradient

            # find end of the peak 
            end_offset = i
            while end_offset < len(smoothed_force) and (smoothed_force[end_offset] > main_threshold or gradient_force[end_offset] <= 0):
                end_offset += 1

            # check if end_offset is out of bounds
            if end_offset >= len(smoothed_force):
                end_offset = len(smoothed_force) - 1

            peaks.append(highest_peak)
            start_rises.append(start_rise)
            end_offsets.append(end_offset)

            # Find subpeaks within this range
            sub_peak_indices, _ = find_peaks(smoothed_force[start_rise:end_offset], height=sub_threshold, distance=distance, prominence=0.02)
            sub_peak_indices = [sp for sp in sub_peak_indices if sp + start_rise != highest_peak]  # exclude main peaks
            sub_peak_indices = [sp + start_rise for sp in sub_peak_indices] 
            sub_peaks.extend(sub_peak_indices)

        else:
            i += 1

    return peaks, start_rises, end_offsets, sub_peaks
