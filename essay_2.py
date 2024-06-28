import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import simpledialog

from emg_processing import *
from prompt_function import *
from readvideo import *
from save_as_json import *
from peakmetrics import *
from PeakDetection import *
from vroom import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib import cm 

plt.ion()

cursor_x, cursor_y = 0, 0 
selected = None 
modified = False

peaks = []
start_of_rise_indices = []
right_bases = []
dlc_peaks = []
dlc_sub_peaks = []

dlc_start_indices = []
dlc_end_indices = []
dlc_sub_peaks_indices = []

root = tk.Tk()
root.withdraw()

def interactive_plot(data_file_path):
    global peaks, start_of_rise_indices, right_bases, time, smoothed_force, force, f_sub_peaks
    global dlc_peaks, dlc_start_indices, dlc_end_indices, dlc_sub_peaks_indices
    global peak_times, start_times, end_times, dlc_sub_peaks, valid_right_bases
    global dlc_amplitudes, dlc_auc_values, dlc_mean_speeds, dlc_max_accelerations, max_speeds, mean_accelerations, durations, displacements, absolute_maxs, dlc_max_amplitudes, dlc_smoothness
    global moving_light_x

    # Robot data
    data = pd.read_csv(data_file_path, sep=" ", header=None)
    if data.shape[1] == 5:
        data.columns = ['Time', 'Force', 'Trigger', 'Reward', 'Adjusted_time']
        time = data['Adjusted_time'].values
    else:
        data.columns = ['Time', 'Force', 'Trigger', 'Reward', 'Adjusted_time', 'calib']
        time = data['Adjusted_time'].values
    force = data['Force'].values


    # EMG
    emg_data_included = False
    answer = simpledialog.askstring("Input", "Do you want to analyze EMG data? (y/n):", parent=root)
    if answer and answer.lower() == 'y':
        emg_file_path = select_emg_file()
        if emg_file_path:
            emg_data_included = True
            time_emg, emg_rectified, emg_env_scaled, fs, threshold, bursts = process_emg_data(emg_file_path)
    
    # DLC
    dlc_file_path = select_DLC_file() 
    if dlc_file_path: 
        results = peak_metrics(dlc_file_path, fps=50, main_threshold=4, sub_threshold=3.5,distance=5, prominence=.5 )
        dlc_peaks, dlc_start_indices, dlc_end_indices, peak_times, start_times, end_times, dlc_sub_peaks,dlc_amplitudes, dlc_auc_values, dlc_mean_speeds, dlc_max_accelerations, max_speeds, mean_accelerations, durations, displacements, absolute_maxs, dlc_sub_peaks, dlc_max_amplitudes, dlc_smoothness = results
        print('Après avoir déposé le fichier', dlc_smoothness, dlc_sub_peaks, dlc_sub_peaks_indices)


    smoothed_force = savgol_filter(force, window_length=31, polyorder=3)
    gradient_force = np.gradient(smoothed_force)
    peaks, start_of_rise_indices, right_bases, f_sub_peaks = f_find_peaks(smoothed_force, gradient_force, main_threshold=0.03, sub_threshold=0.015, distance=25)
    
    colors = cm.Paired(np.linspace(0, 1, len(peaks)))
    valid_right_bases = [idx for idx in right_bases if idx < len(smoothed_force)]

    num_plots = 3 if emg_data_included else 2
    fig, axes = plt.subplots(num_plots, 1, figsize=[12, 12 * num_plots // 2])
    ax1, ax2 = axes[0], axes[1]

    # plt force
    ax1.plot(time, force, label='Raw force', alpha=0.5)
    ax1.plot(time, smoothed_force, label='Smoothed force', color='DarkBlue')
    ax1.hlines(0.03, 0, time[-1])
    ax1.set_ylim([-.2, .4])

    peak_dots, = ax1.plot(time[peaks], smoothed_force[peaks], 'ro', markersize=8, label='Peaks')
    start_dots, = ax1.plot(time[start_of_rise_indices], smoothed_force[start_of_rise_indices], 'go', markersize=8, label='Start of rise')
    end_dots, = ax1.plot(time[valid_right_bases], smoothed_force[valid_right_bases], 'bo', markersize=8, label='End of rise')

    for sub_peak in f_sub_peaks:
        if sub_peak < len(smoothed_force):
            ax1.scatter(data['Time'].iloc[sub_peak], smoothed_force[sub_peak], color='red', zorder=2.5, marker='x')
    ax1.set_ylabel('Force')

    peak_indices = (np.array(peak_times) * 50).astype(int)
    dlc_peaks = peak_indices.tolist()
    dlc_start_indices = (np.array(start_times) * 50).astype(int).tolist()
    dlc_end_indices = (np.array(end_times) * 50).astype(int).tolist()
    dlc_sub_peaks_indices = (np.array(dlc_sub_peaks) * 50).astype(int).tolist()
    print(dlc_sub_peaks_indices)

    # Plot DLC 
    moving_light_x = load_data(dlc_file_path)
    fps = 50
    ax2.plot(np.arange(len(moving_light_x)) / fps, moving_light_x, label='Moving light', color='gray')
    valid_dlc_peaks = [idx for idx in dlc_peaks if idx < len(moving_light_x)]
    valid_dlc_start_indices = [idx for idx in dlc_start_indices if idx < len(moving_light_x)]
    valid_dlc_end_indices = [idx for idx in dlc_end_indices if idx < len(moving_light_x)]
    valid_sub_peaks = [idx for idx in dlc_sub_peaks_indices if idx < len(moving_light_x)]
    print(valid_sub_peaks)

    dlc_peak_dots, = ax2.plot(np.array(valid_dlc_peaks) / fps, moving_light_x[valid_dlc_peaks], 'ro', markersize=6, label='DLC Peaks')
    dlc_start_dots, = ax2.plot(np.array(valid_dlc_start_indices) / fps, moving_light_x[valid_dlc_start_indices], 'go', markersize=6, label='DLC Peak starts')
    dlc_end_dots, = ax2.plot(np.array(valid_dlc_end_indices) / fps, moving_light_x[valid_dlc_end_indices], 'bo', markersize=5, label='DLC Peak ends')
    dlc_sub_peak_dots, = ax2.plot(np.array(dlc_sub_peaks) / fps, moving_light_x[dlc_sub_peaks], 'rx',markersize=6, label='Sub Peaks')

    #ax2.legend()
    ax2.grid(True)

    # plot EMG if included
    if emg_data_included:
        ax3 = axes[2]
        ax3.plot(time_emg, emg_rectified, label='Rectified emg', alpha=0.5)
        ax3.plot(time_emg, emg_env_scaled, label='EMG scaled envelope', linewidth=2)
        ax3.hlines(threshold, time_emg[0], time_emg[-1], color='black', label='Threshold')
        for burst in bursts:
            ax3.scatter(burst['start_time'], threshold, color='black')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('EMG signal amplitude (mV)')
        ax3.legend()

    plt.tight_layout()
    plt.show(block=False)

    # keys thinnngy
    def on_key_press(event):
        global peaks, start_of_rise_indices, cursor_x, right_bases, f_sub_peaks, dlc_peaks, dlc_start_indices, dlc_end_indices, dlc_sub_peaks_indices, modified, valid_right_bases
        modified = True

        target_peaks = None
        target_starts = None
        target_ends = None
        target_sub_peaks = None
        target_time = None
        target_data = None

        if event.inaxes == ax1:
            target_peaks = peaks
            target_starts = start_of_rise_indices
            target_ends = right_bases
            target_sub_peaks = f_sub_peaks
            target_time = time
            target_data = smoothed_force
        elif event.inaxes == ax2:
            target_peaks = dlc_peaks
            target_starts = dlc_start_indices
            target_ends = dlc_end_indices
            target_sub_peaks = dlc_sub_peaks_indices
            target_time = np.arange(len(moving_light_x)) / fps
            target_data = moving_light_x

        if target_peaks is None:
            return

        if event.key == 'z':  # Delete nearest peak
            if len(target_peaks) > 0:
                distances = np.abs(target_time[target_peaks] - cursor_x)
                nearest_peak_idx = np.argmin(distances)
                target_peaks.pop(nearest_peak_idx)

        elif event.key == 'v':  # Delete start of rise
            if len(target_starts) > 0:
                distances = np.abs(target_time[target_starts] - cursor_x)
                nearest_start_idx = np.argmin(distances)
                target_starts.pop(nearest_start_idx)

        elif event.key == 'a':  # Add a new peak
            new_peak_idx = np.argmin(np.abs(target_time - cursor_x))
            target_peaks.append(new_peak_idx)

        elif event.key == 'g':  # Add a new start of rise
            new_rise_idx = np.argmin(np.abs(target_time - cursor_x))
            target_starts.append(new_rise_idx)

        elif event.key == 'b':  # Add a new end of rise
            new_end_idx = np.argmin(np.abs(target_time - cursor_x))
            target_ends.append(new_end_idx)
            print('Added new end of rise at index:', new_end_idx)

        elif event.key == 'n':  # Delete nearest end
            if len(target_ends) > 0:
                distances = np.abs(target_time[target_ends] - cursor_x)
                nearest_end_idx = np.argmin(distances)
                target_ends.pop(nearest_end_idx)
            print('Deleted nearest end.')

        elif event.key == 'p':
            print("Number of peaks:", len(target_peaks))
            print("Start of rise indices:", len(target_starts))
            print("End of peaks:", len(target_ends))
            
        update_plot()

    def on_mouse_move(event):
        global cursor_x, cursor_y
        if event.inaxes:
            cursor_x, cursor_y = event.xdata, event.ydata
        else:
            cursor_x, cursor_y = None, None

    def update_plot():
        global peaks, start_of_rise_indices, right_bases, dlc_peaks, dlc_start_indices, dlc_end_indices, dlc_sub_peaks_indices, dlc_sub_peaks, valid_right_bases
        
        valid_right_bases = [idx for idx in right_bases if idx < len(smoothed_force)]

        valid_dlc_peaks = [idx for idx in dlc_peaks if idx < len(moving_light_x)]
        valid_dlc_start_indices = [idx for idx in dlc_start_indices if idx < len(moving_light_x)]
        valid_dlc_end_indices = [idx for idx in dlc_end_indices if idx < len(moving_light_x)]
        valid_sub_peaks = [idx for idx in dlc_sub_peaks if idx < len(moving_light_x)]

        peak_dots.set_data(time[peaks], smoothed_force[peaks])
        start_dots.set_data(time[start_of_rise_indices], smoothed_force[start_of_rise_indices])
        end_dots.set_data(time[valid_right_bases], smoothed_force[valid_right_bases])

        dlc_peak_dots.set_data(np.array(valid_dlc_peaks) / fps, moving_light_x[valid_dlc_peaks])
        dlc_start_dots.set_data(np.array(valid_dlc_start_indices) / fps, moving_light_x[valid_dlc_start_indices])
        dlc_end_dots.set_data(np.array(valid_dlc_end_indices) / fps, moving_light_x[valid_dlc_end_indices])
        dlc_sub_peak_dots.set_data(np.array(dlc_sub_peaks) / fps, moving_light_x[dlc_sub_peaks])

        fig.canvas.draw_idle()

    selected_scatter = None
    selected_index = None
    selected_data = None

    def on_press(event):
        nonlocal selected_scatter, selected_index, selected_data
        if event.inaxes not in [ax1, ax2]:
            return
        for scatter, data in [(peak_dots, peaks), (start_dots, start_of_rise_indices), (end_dots, right_bases),
                          (dlc_peak_dots, dlc_peaks), (dlc_start_dots, dlc_start_indices), (dlc_end_dots, dlc_end_indices),
                          (dlc_sub_peak_dots, dlc_sub_peaks)]:
                          
            contains, index = scatter.contains(event)
            if contains:
                selected_scatter = scatter
                selected_index = index['ind'][0]
                selected_data = data
                break

    def on_release(event):
        nonlocal selected_scatter, selected_index, selected_data
        selected_scatter = None
        selected_index = None
        selected_data = None

    def on_motion(event):
        global modified

        if selected_scatter is None:
            return
        x, y = event.xdata, event.ydata
        if x is not None:
            if selected_scatter in [dlc_peak_dots, dlc_start_dots, dlc_end_dots, dlc_sub_peak_dots]:
                selected_data[selected_index] = np.argmin(np.abs(np.arange(len(moving_light_x)) - x * fps))
            else:
                selected_data[selected_index] = np.argmin(np.abs(time - x))
            modified = True
            update_plot()

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    while True:
        play_video_decision = ask_watch_video()
        if not play_video_decision:
            break
    
        video_time_ranges = get_video_segment()
        if video_time_ranges:
            video_path = select_video_file()
            if video_path:
                play_video_segment(video_path, *video_time_ranges)

def calculate_amplitudes(smoothed_force, peaks, start_of_rise_indices):
    amplitudes = []
    for peak_idx, start_idx in zip(peaks, start_of_rise_indices):
        amplitude = smoothed_force[peak_idx] - smoothed_force[start_idx]
        amplitudes.append(amplitude)
    return amplitudes

def save_data_to_json():
    global peaks, start_of_rise_indices, modified, time, smoothed_force, force, right_bases, f_sub_peaks, force_smoothness
    global dlc_amplitudes, dlc_auc_values, dlc_mean_speeds, dlc_max_accelerations, max_speeds, mean_accelerations, durations, displacements, absolute_maxs, dlc_max_amplitudes, dlc_smoothness
    global peak_times, start_times, end_times, dlc_sub_peaks, dlc_peaks, dlc_start_indices, dlc_end_indices
    global moving_light_x

    if modified:  # save and calculate metrics if there have been modifications
        peak_metrics = calculate_peak_metrics(time, smoothed_force, start_of_rise_indices, peaks, right_bases, f_sub_peaks)
        AUCurve = [metric['AUC'] for metric in peak_metrics]
        peak_duration = [metric['Peak duration'] for metric in peak_metrics]
        mean_amplitude = [metric['Mean amplitude'] for metric in peak_metrics]
        max_amplitude = [metric['Max amplitude'] for metric in peak_metrics]
        f_sub_peaks = [metric['f_sub_peaks'] for metric in peak_metrics]
        force_smoothness = [metric['force_smoothness'] for metric in peak_metrics]

        # Calling metrics_DLC again for recalculation
        dlc_peaks, dlc_start_indices, dlc_end_indices, peak_times, start_times, end_times,dlc_sub_peaks, dlc_amplitudes, dlc_auc_values, dlc_mean_speeds, dlc_max_accelerations, max_speeds, mean_accelerations, durations, displacements, absolute_maxs, dlc_max_amplitudes, dlc_smoothness = metrics_DLC(
            moving_light_x, dlc_peaks, dlc_start_indices, dlc_end_indices, dlc_sub_peaks, fps=50)

        modified = False
    else:
        peak_metrics = calculate_peak_metrics(time, smoothed_force, start_of_rise_indices, peaks, right_bases, f_sub_peaks)
        AUCurve = [metric['AUC'] for metric in peak_metrics]
        peak_duration = [metric['Peak duration'] for metric in peak_metrics]
        mean_amplitude = [metric['Mean amplitude'] for metric in peak_metrics]
        max_amplitude = [metric['Max amplitude'] for metric in peak_metrics]
        f_sub_peaks = [metric['f_sub_peaks'] for metric in peak_metrics]
        force_smoothness = [metric['force_smoothness'] for metric in peak_metrics]


        dlc_peaks, dlc_start_indices, dlc_end_indices, peak_times, start_times, end_times,dlc_sub_peaks, dlc_amplitudes, dlc_auc_values, dlc_mean_speeds, dlc_max_accelerations, max_speeds, mean_accelerations, durations, displacements, absolute_maxs, dlc_max_amplitudes, dlc_smoothness = metrics_DLC(
            moving_light_x, dlc_peaks, dlc_start_indices, dlc_end_indices, dlc_sub_peaks, fps=50)

        modified = False

    peak_coordinates = [{'time': time[p], 'force': smoothed_force[p]} for p in peaks]
    start_of_rise_coordinates = [{'time': time[s], 'force': smoothed_force[s]} for s in start_of_rise_indices]
    right_bases_coordinates = [{'time': time[r], 'force': smoothed_force[r]} for r in right_bases]

    f_amplitudes = calculate_amplitudes(smoothed_force, peaks, start_of_rise_indices)
    peak_count = len(peaks)
    t_duration = time[-1] - time[0] 

    adjusted_data_file = specify_json_file_path()
    if adjusted_data_file:
        save_adjusted_data(f_amplitudes, AUCurve, peak_duration, mean_amplitude, max_amplitude, f_sub_peaks, force_smoothness, peak_count,
                           dlc_amplitudes, dlc_auc_values, dlc_mean_speeds, dlc_max_accelerations,
                           t_duration, peaks, start_of_rise_indices, right_bases_coordinates,
                           peak_coordinates, start_of_rise_coordinates, peak_times, start_times, end_times, dlc_sub_peaks, smoothed_force, force,
                           durations, displacements, absolute_maxs, max_speeds, mean_accelerations, dlc_max_amplitudes, dlc_smoothness, adjusted_data_file)

        print("Data saved to JSON file.")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    data_file_path = select_data_file() 
    interactive_plot(data_file_path)
    decision = simpledialog.askstring("Save", "Do you want to save the changes? (y/n):", parent=root)
    if decision and decision.lower() == 'y':
        save_data_to_json()
