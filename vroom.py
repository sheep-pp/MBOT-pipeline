import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import trapz
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pd.read_csv(file_path, header=2)
    moving_light_x = pd.to_numeric(data['x.1'], errors='coerce')
    max_value = moving_light_x.max()
    normalized_data = max_value - moving_light_x
    return normalized_data

def calculate_speed_and_acceleration(data):
    speed = np.gradient(data)
    acceleration = np.gradient(speed)
    return speed, acceleration

def d_find_peaks(smoothed_force, gradient_force, main_threshold=1.5, sub_threshold=1, distance=25, prominence=0.02):
    peaks = []  
    start_rises = [] 
    end_offsets = []  
    sub_peaks = [] 

    i = 0
    while i < len(smoothed_force):
        if smoothed_force[i] > main_threshold and gradient_force[i] >= 0:
            highest_peak = i
            while i < len(smoothed_force) and (smoothed_force[i] > main_threshold or gradient_force[i] >= 0):
                if smoothed_force[i] > smoothed_force[highest_peak]:
                    highest_peak = i
                i += 1

            start_rise = highest_peak
            while start_rise > 0 and not (smoothed_force[start_rise] < main_threshold and gradient_force[start_rise] < 0):
                start_rise -= 1
            start_rise += 1

            end_offset = i
            while end_offset < len(smoothed_force) and (smoothed_force[end_offset] > main_threshold or gradient_force[end_offset] <= 0):
                end_offset += 1

            if end_offset >= len(smoothed_force):
                end_offset = len(smoothed_force) - 1

            peaks.append(highest_peak)
            start_rises.append(start_rise)
            end_offsets.append(end_offset)

            sub_peak_indices, _ = find_peaks(smoothed_force[start_rise:end_offset], height=sub_threshold, distance=distance, prominence=prominence)
            sub_peak_indices = [sp for sp in sub_peak_indices if sp + start_rise != highest_peak]
            sub_peak_indices = [sp + start_rise for sp in sub_peak_indices]
            sub_peaks.extend(sub_peak_indices)

            print(f"Start: {start_rise}, Peak: {highest_peak}, End: {end_offset}, Sub-peaks: {sub_peak_indices}")

        else:
            i += 1
    return peaks, start_rises, end_offsets, sub_peaks

def find_peak_starts(data, DLCpeaks, fps, start_offset=10, y_threshold=2):
    gradients = np.gradient(data)
    gradient_threshold = np.median(np.abs(gradients))
    peak_starts = []
    start_times = []

    for peak in DLCpeaks:
        potential_start = max(0, peak - start_offset)
        found = False
        for i in range(potential_start, peak):
            if gradients[i] < -gradient_threshold:
                peak_starts.append(i)
                start_times.append(i / fps)  
                found = True
                break

        if not found:
            peak_starts.append(potential_start)
            start_times.append(potential_start / fps)  

    refined_starts = []
    refined_start_times = [] 
    for index, start in enumerate(peak_starts):
        if data[start] > y_threshold:
            i = start - 1
            while i > 0 and not (data[i] < y_threshold and gradients[i] < -gradient_threshold):
                i -= 1
            refined_starts.append(i)
            refined_start_times.append(i / fps) 
        else:
            refined_starts.append(start)
            refined_start_times.append(start_times[index])  

    return refined_starts, refined_start_times

def find_peak_ends(data, DLCpeaks, fps, forward_frames=2, y_threshold=2):
    gradients = np.gradient(data)
    gradient_threshold = np.median(np.abs(gradients))
    peak_ends = []
    end_times = []

    for peak in DLCpeaks:
        potential_end = peak
        while potential_end < len(data) - 1 and gradients[potential_end] > -gradient_threshold:
            potential_end += 1
        
        peak_ends.append(potential_end)  
        end_times.append(potential_end / fps) 

    refined_ends = []
    refined_end_times = [] 

    for index, end in enumerate(peak_ends):
        if data[end] > y_threshold:
            i = end + forward_frames
            while i < len(data) - 1 and not (data[i] < y_threshold and gradients[i] > -gradient_threshold):
                i += 1
            refined_ends.append(i)
            refined_end_times.append(i / fps)  
        else:
            refined_ends.append(end)
            refined_end_times.append(end_times[index])  

    return refined_ends, refined_end_times

def calculate_peak_times(peaks, fps):
    return [peak / fps for peak in peaks]


def metrics_DLC(data, DLCpeaks, peak_starts, peak_ends, sub_peaks, fps):
    amplitudes, auc_values, mean_speeds, max_accelerations, max_speeds, mean_accelerations, durations, displacements, absolute_maxs, dlc_max_amplitudes, dlc_smoothness = [], [], [], [], [], [], [], [], [], [], []
    speed, acceleration = calculate_speed_and_acceleration(data)

    print("Starting metrics_DLC calculations...")
    for peak, start, end in zip(DLCpeaks, peak_starts, peak_ends):
        if start < 0 or peak >= len(data) or end >= len(data):
            print(f"Skipping invalid indices: start={start}, peak={peak}, end={end}")
            continue
        if start >= peak:
            print(f"Skipping because start >= peak: start={start}, peak={peak}")
            continue

        amplitude = data[peak] - data[start]
        amplitudes.append(amplitude)

        auc = trapz(data[start:end+1], dx=1/fps)
        auc_values.append(auc)

        mean_speed = np.mean(speed[start:peak+1]) if peak > start else np.nan
        mean_speeds.append(mean_speed)

        if peak > start:
            max_accel = np.max(acceleration[start:peak+1]) if peak > start else np.nan
            max_accelerations.append(max_accel)

            max_speed = np.max(speed[start:peak+1]) if peak > start else np.nan
            max_speeds.append(max_speed)

            mean_acceleration = np.mean(acceleration[start:peak+1]) if peak > start else np.nan
            mean_accelerations.append(mean_acceleration)
        else:
            max_accelerations.append(np.nan)
            max_speeds.append(np.nan)
            mean_accelerations.append(np.nan)

        duration = (end - start) / fps if end > start else np.nan
        durations.append(duration)

        displacement = np.sum(np.abs(np.diff(data[start:end+1]))) if end > start else 0
        displacements.append(displacement)

        absolute_max = data[peak]
        absolute_maxs.append(absolute_max)

        # Smoothness
        v_sub_peaks = sum(sp >= start and sp <= end for sp in sub_peaks)
        a_dlc_smoothness = 1 / (v_sub_peaks + 1)
        dlc_smoothness.append(a_dlc_smoothness)

        # DLC max amplitudes
        sub_peaks_in_range = [sp for sp in sub_peaks if start <= sp <= end]
        a = np.sum([data[sp] - data[start] for sp in sub_peaks_in_range])
        b = data[peak] - data[start]
        dlc_max_amplitude = (a + b) / (len(sub_peaks_in_range) + 1) if len(sub_peaks_in_range) > 0 else b
        dlc_max_amplitudes.append(dlc_max_amplitude)

    peak_times = calculate_peak_times(DLCpeaks, fps)
    start_times = calculate_peak_times(peak_starts, fps)
    end_times = calculate_peak_times(peak_ends, fps)

    print("Finished metrics_DLC calculations.")
    print(f"Final dlc_smoothness: {dlc_smoothness}")
    return DLCpeaks, peak_starts, peak_ends, peak_times, start_times, end_times, sub_peaks, amplitudes, auc_values, mean_speeds, max_accelerations, max_speeds, mean_accelerations, durations, displacements, absolute_maxs, dlc_max_amplitudes, dlc_smoothness


def mon_cul(file_path, fps=50, main_threshold=1, sub_threshold=0.5, distance=5, prominence=1):
    moving_light_x = load_data(file_path)
    gradient_data = np.gradient(moving_light_x)
    
    print(f"Analyzing file: {file_path}")
    print(f"Parameters - main_threshold: {main_threshold}, sub_threshold: {sub_threshold}, distance: {distance}, prominence: {prominence}")
    
    DLCpeaks, peak_starts, peak_ends, sub_peaks = d_find_peaks(moving_light_x, gradient_data, main_threshold, sub_threshold, distance, prominence)
    
    DLCpeaks, peak_starts, peak_ends, peak_times, start_times, end_times,sub_peaks, amplitudes, auc_values, mean_speeds, max_accelerations, max_speeds, mean_accelerations, durations, displacements, absolute_maxs, dlc_max_amplitudes, dlc_smoothness = metrics_DLC(moving_light_x, DLCpeaks, peak_starts, peak_ends, sub_peaks, fps)

    return DLCpeaks, peak_starts, peak_ends, peak_times, start_times, end_times,sub_peaks, amplitudes, auc_values, mean_speeds, max_accelerations, max_speeds, mean_accelerations, durations, displacements, absolute_maxs, sub_peaks, dlc_max_amplitudes, dlc_smoothness
