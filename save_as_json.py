import numpy as np
import json
from collections import OrderedDict

def save_adjusted_data(f_amplitudes, AUC, force_duration, mean_amplitude, max_amplitude, sub_peaks, force_smoothness, peak_count, 
                        dlc_amplitudes, dlc_auc_values, dlc_mean_speeds, dlc_max_accelerations, 
                        task_duration, peaks, start_of_rise_indices, right_bases_coordinates,peak_coordinates,
                        start_of_rise_coordinates, peak_times, start_times, end_times, dlc_sub_peaks, smoothed_force, force, 
                        durations,displacements,absolute_maxs, max_speeds, mean_accelerations,dlc_max_amplitudes,dlc_smoothness,
                        file_path):#,file_2_path):

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj
    print("Convert type before use:", type(convert))

    data_to_save = OrderedDict([
        ('force_max_peak_amplitude', convert(f_amplitudes)),
        ('force_AUC', convert(AUC)),
        ('force_peak_duration', convert(force_duration)),
        ('force_mean_amplitude',convert(mean_amplitude)),
        ('force_max_p_sp_amplitudes',convert(max_amplitude)),
        ('Force_freq_peaks', convert(sub_peaks)), #number of sub_peaks in one peak
        ('force_smoothness', convert(force_smoothness)),

        ('force_number of peaks', convert(peak_count)),
        ('task_duration', convert(task_duration)),


# attention : dlc_amplitudes : rouge - vert tandis que absolute_maxs = rouge dans les fonctions DLC. dans le json et comment ils sont nommées absolutes_maxs n'est pas pris en compte et ce qu'on nomme absolutes max c'est rouge - vert.
        #('dlc_max_amplitudes', convert(dlc_amplitudes)), # a enlever 
        ('dlc_absolute_maxs',convert(dlc_amplitudes)),
        ('dlc_max_p_sp_amplitudes', convert(dlc_max_amplitudes)), #plrs amplitudes/sommes des peaks
        ('dlc_AUC', convert(dlc_auc_values)),
        ('dlc_mean_speeds', convert(dlc_mean_speeds)),
        ('dlc_max_speeds', convert(max_speeds)),
        ('dlc_max_accelerations', convert(dlc_max_accelerations)),
        ('dlc_mean_accelerations', convert(mean_accelerations)),
        ('dlc_duration', convert(durations)),
        ('displacements',convert(displacements)),
        #('dlc_sub_peaks',convert(dlc_sub_peaks)),
        ('dlc_smoothness', convert(dlc_smoothness)),

        #dlc
        ('DLC_peak_times', convert(peak_times)),
        ('DLC_start_times', convert(start_times)),
        ('DLC_end_times', convert(end_times)),
#force
        ('peaks_indice', convert(peaks)),
        ('start_of_rise_indices', convert(start_of_rise_indices)),

#force
        ('peak_coordinates', convert(peak_coordinates)),
        ('start_of_rise_coordinates', convert(start_of_rise_coordinates)),
        ('end_of_rise_coordinates', convert(right_bases_coordinates)),

        ('smoothen_force', convert(smoothed_force)),
        ('raw_force', convert(force))    
    ])


    with open(file_path, 'w') as f:
        json.dump(data_to_save, f, default=convert, indent=4, sort_keys=False)

   #with open (file_2_path,'w') as f:
        #json.dump(data_to_save,f,default=convert, indent=4, sort_keys=False)

    