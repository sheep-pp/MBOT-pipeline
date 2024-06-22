import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.signal import filtfilt, butter
from scipy.integrate import trapz

def rms_signal(signal, window_size):
    # window size to be odd pour que ce soit symmetrique 
    if window_size % 2 == 0:
        window_size += 1
    conv_window = np.ones(window_size) / window_size
    rms = np.sqrt(np.convolve(signal ** 2, conv_window, 'same'))
    return rms

'''def moving_average_filter(signal, window_size):
    """Apply a moving average filter to the signal."""
    if window_size % 2 == 0:  # Ensure window size is odd
        window_size += 1
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal, window, mode='same')
    return smoothed_signal'''

def process_emg_data(emg_file_path, start_index=None, end_index=None):
    # EMG data
    data = loadmat(emg_file_path)  
    emg_data_full = data['data'].squeeze()  
    fs = data['samplerate'][0][0]   
    
    if start_index is None:
        start_index = 0
    if end_index is None or end_index > len(emg_data_full):
        end_index = len(emg_data_full)

    emg_data = emg_data_full[start_index:end_index]
    time_emg = np.arange(start_index, end_index) / fs
    
    # val abs 
    emg_rectified = np.abs(emg_data)

    # Apply RMS to rectified signal
    window_size_rms = int(fs * 0.05)  # Define RMS window size (e.g., 50 ms)
    emg_rms = rms_signal(emg_rectified, window_size_rms)
    
    #emg_smoothed = moving_average_filter(emg_rms, window_size_ma)

    peak_rectified = np.max(emg_rectified) #sur + signal
    peak_rms = np.max(emg_rms)
    #peak_env = np.max(emg_smoothed) #sur petit rms
    scaling_factor = peak_rectified / peak_rms
    emg_env_scaled = emg_rms * scaling_factor

    emg_median = np.median(emg_data)
    emg_std = np.std(emg_data)
    threshold = emg_median + 2 * emg_std

    above_thresh = np.where(emg_env_scaled > threshold)[0]
    if above_thresh.size == 0:
        print("No bursts detected.")

#  the start and end indices of bursts
    gaps = np.diff(above_thresh) > 10000
    start_indices = np.insert(above_thresh[np.where(gaps)[0] + 1], 0, above_thresh[0])
    end_indices = np.append(above_thresh[np.where(gaps)[0]], above_thresh[-1])
    start_indices_time = start_indices/fs
    end_indices_time = end_indices/fs

    bursts = []
    for start, end in zip(start_indices, end_indices):
        burst_duration = (end - start) / fs
        burst_mean = np.mean(emg_env_scaled[start:end])
        burst_max = np.max(emg_env_scaled[start:end])
        burst_auc = np.trapz(emg_env_scaled[start:end]) / burst_duration
            #to add in the json file 
        bursts.append({'start_time': start / fs, 'mean': burst_mean, 
                       'max': burst_max, 'duration': burst_duration, 'auc': burst_auc})
        
        print(f"Burst start at {start/fs}s, Mean={burst_mean}, Max={burst_max}, Duration={burst_duration}s, AUC={burst_auc}")
   
    return time_emg, emg_rectified, emg_env_scaled, fs, threshold, bursts
