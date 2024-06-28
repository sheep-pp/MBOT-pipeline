import json
import pandas as pd
import os

'''in this code you have to change 3 pathnames
1) directory where json are located
2) output file where table with raw data are saved
3) output file where table with normalized data are saved'''



def parse_filename(filename):
    """
    Parses the filename to extract the day, animal number, and condition.
    Expected format: 'DayDPI_AnimalID_Condition.json' except for Baseline where it is : 'BL_AnimalID.json'
    ex filename: '2WPI_M01_STIM_10KHz.json
    'BL_M15.json'
    Returns:
        condition (str): The condition from the filename.
        animal_id (str): The extracted animal ID.
        day (int or str): The day number extracted and converted to an integer, or 'BL'
    """
    parts = filename.replace('.json', '').split('_')
    
    if len(parts) < 3:
        day = 'BL'
        animal_id = parts[1] if len(parts) > 1 else 'unknown'
        condition = 'wo'
    else:
        day_part = parts[0].replace('DPI', '')
        try:
            day = int(day_part) 
        except ValueError:
            day = day_part 

        animal_id = parts[1]
        if 'STIM' in parts[2]:
            condition = parts[2] + '_' + parts[3]
        else:
            condition = parts[2]
    
    return day, animal_id, condition

# where all the json are located
directory = '/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/video_test_code_v2/test_emg/azerty'


mean_values_list = []

for filename in os.listdir(directory):
    if filename.endswith('.json'):
        file_path = os.path.join(directory, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file {filename}")
            continue

      
        means = {}
        for param, values in data.items():
            if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                means[param] = sum(values) / len(values)
            elif isinstance(values, (int, float)):
                means[param] = values

        #  frequency of pull
        if 'force_number of peaks' in data and 'task_duration' in data:
            force_number_of_peaks = data['force_number of peaks']
            task_duration = data['task_duration']
            if task_duration != 0:
                frequency_of_pull = force_number_of_peaks / task_duration
            else:
                frequency_of_pull = 0
            means['frequency_of_pull'] = frequency_of_pull

        # 5 highest force peaks
        if 'force_max_peak_amplitude' in data and isinstance(data['force_max_peak_amplitude'], list):
            highest_peaks = sorted(data['force_max_peak_amplitude'], reverse=True)[:5]
            if highest_peaks:
                mean_highest_peaks = sum(highest_peaks) / len(highest_peaks)
            else:
                mean_highest_peaks = 0
            means['5highest_force_peak'] = mean_highest_peaks

        try:
            
            day, animal_id, condition = parse_filename(filename)
        except ValueError as e:
            print(e)
            continue

        means['day'] = day
        means['animal_id'] = animal_id
        means['condition'] = condition

        mean_values_list.append(means)


df = pd.DataFrame(mean_values_list)

# exlude columns
exclude_columns = [
    'DLC_peak_times', 'DLC_start_times', 'DLC_end_times', 
    'peaks_indice', 'start_of_rise_indices', 
    'raw_force', 'smoothen_force', 'task_duration'
]
df = df[[col for col in df.columns if col not in exclude_columns]]

# save raw 
raw_output_file = '/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/mean_files.csv'
df.to_csv(raw_output_file, index=False)
print(f"raw mean values have been saved to {raw_output_file}")

bl_values = df[df['day'] == 'BL'].set_index('animal_id')

# normalization
normalized_data = []
for _, row in df.iterrows():
    animal_id = row['animal_id']
    if animal_id in bl_values.index:
        bl_row = bl_values.loc[animal_id]
        normalized_row = row.copy()
        for col in row.index:
            if col not in ['day', 'animal_id', 'condition'] and col in bl_row.index:
                normalized_row[col] = row[col] / bl_row[col]
        normalized_data.append(normalized_row)

normalized_df = pd.DataFrame(normalized_data)

# Save normalized 
normalized_output_file = '/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/normalized_mean_files.csv'
normalized_df.to_csv(normalized_output_file, index=False)

print(f"normalized mean values have been saved to {normalized_output_file}")