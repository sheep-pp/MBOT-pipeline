library(dplyr)
library(tidyr)



#this code create the csv files to do the statistics for prims  of indiv param
# change the output_file pathname

csv_file <- '/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/normalized_mean_files.csv'
df <- read.csv(csv_file)

# Filter
filtered_df <- df %>%
  filter(day %in% c('BL', '1WPI', '2WPI', '3WPI', '4WPI'),
         condition %in% c('wo', 'sham', 'STIM_10KHz', 'STIM_40Hz'),
         animal_id %in% c('M02', 'M03', 'M07', 'M09', 'M10', 'M13', 'M12')) %>%
  mutate(day = factor(day, levels = c('BL', '1WPI', '2WPI', '3WPI', '4WPI')))

exclude_columns <- c('DLC_peak_times', 'DLC_start_times', 'DLC_end_times', 
                     'peaks_indice', 'start_of_rise_indices', 
                     'raw_force', 'smoothen_force', 'task_duration')

numerical_columns <- sapply(filtered_df, is.numeric)
columns_to_include <- names(numerical_columns[numerical_columns == TRUE])
columns_to_include <- setdiff(columns_to_include, exclude_columns)

prism_df <- filtered_df %>%
  select(animal_id, day, condition, all_of(columns_to_include))

output_file <- "/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/figure_nulle/plot_R/both_stim_LH_individual_parameters_for_prism.csv"
write.csv(prism_df, output_file, row.names = FALSE)

head(prism_df)
