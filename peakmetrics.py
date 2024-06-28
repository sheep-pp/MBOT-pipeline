import numpy as np

def calculate_peak_metrics(time, smoothed_force, start_of_rise_indices, peaks, right_bases, sub_peaks):
    if isinstance(sub_peaks, np.integer):
        sub_peaks = [sub_peaks]
    
    # Convert sub_peaks to integers and filter out invalid indices
    sub_peaks = [int(sp) for sp in sub_peaks if 0 <= int(sp) < len(smoothed_force)]
    
    metrics = []
    for start, peak, end in zip(start_of_rise_indices, peaks, right_bases):
        if end >= start:  # check the slice is non-empty
            peak_range = smoothed_force[start:end+1]
            time_range = time[start:end+1]

            abs_peak_range = np.abs(peak_range)
            # AUC for each peak
            auc = np.trapz(abs_peak_range, time_range) if end+1 > start else 0
            
            # Duration of each peak
            duration = time[end] - time[start] if end > start else 0
            
            # Mean and max amplitude of each peak
            mean_amplitude = np.mean(peak_range) if end+1 > start else 0
            #max_amplitude_standard = np.max(peak_range) if end+1 > start else 0

            # nulber of peaks (number of sub_peaks +1 (corresponds to the peak) over the peak duration)
            a_sub_peaks = sum(start <= sp <= end for sp in sub_peaks)
            num_sub_peaks = (a_sub_peaks + 1) / duration if duration > 0 else 0  
            #print('force sub_peak : a_sub_peaks', a_sub_peaks)
            
            # Force smoothness
            force_smoothness = 1 / (a_sub_peaks + 1)

            # Custom metric
            sub_peaks_in_range = [sp for sp in sub_peaks if start <= sp <= end]
            #print(f"Sub peaks in range: {sub_peaks_in_range}")

            a = np.sum([smoothed_force[sp] - smoothed_force[start] for sp in sub_peaks_in_range])
            b = smoothed_force[peak] - smoothed_force[start]
            max_amplitude = (a + b) / (len(sub_peaks_in_range) + 1) if len(sub_peaks_in_range) > 0 else b
            
            #print('f_sb',sub_peaks_in_range, a, b, max_amplitude)

            metrics.append({
                "AUC": auc,
                "Peak duration": duration,
                "Mean amplitude": mean_amplitude,
                "Max amplitude": max_amplitude,
                "f_sub_peaks": num_sub_peaks,
                "force_smoothness": force_smoothness
            })
        else:
            # case where nothing is found (when start and end are not correct)
            metrics.append({
                "AUC": 0,
                "Peak duration": 0,
                "Mean amplitude": 0,
                "Max amplitude": 0,
                "f_sub_peaks": 0, 
                "force_smoothness": 0
            })

    return metrics
