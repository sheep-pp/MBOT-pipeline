library(dplyr)
library(tidyr)


#this code create the csv files to do the statistics for prims of pc scores
# change the output_file pathname

csv_file <- '/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/normalized_mean_files.csv'
df <- read.csv(csv_file)

# Filter
filtered_df <- df %>%
  filter(day %in% c('BL', '1WPI', '2WPI', '3WPI', '4WPI'),
         condition %in% c( 'STIM_40Hz'),
         animal_id %in% c( 'M07', 'M02', 'M03', 'M09', 'M12', 'M13', 'M10')) %>%
  mutate(day = factor(day, levels = c('BL', '1WPI', '2WPI', '3WPI', '4WPI')))

exclude_columns <- c('DLC_peak_times', 'DLC_start_times', 'DLC_end_times', 
                     'peaks_indice', 'start_of_rise_indices', 
                     'raw_force', 'smoothen_force', 'task_duration')

numerical_columns <- sapply(filtered_df, is.numeric)
columns_to_include <- names(numerical_columns[numerical_columns == TRUE])
columns_to_include <- setdiff(columns_to_include, exclude_columns)

# Standardize the data
scaled_data <- scale(select(filtered_df, all_of(columns_to_include)))
pca <- prcomp(scaled_data, center = TRUE, scale. = TRUE)
pca_df <- as.data.frame(pca$x)
pca_df$Animal <- filtered_df$animal_id
pca_df$Day <- filtered_df$day
pca_df$Condition <- filtered_df$condition

# save df for Prism
output_file <- "/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/figure_nulle/plot_R/pca_scores_both_STIM.csv"
write.csv(pca_df, output_file, row.names = FALSE)

pca_df