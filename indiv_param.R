library(dplyr)
library(ggplot2)


# generate indiv param and save them in the a directory to change yourself

csv_file <- '/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/normalized_mean_files.csv'
df <- read.csv(csv_file)

# Filter 
filtered_df <- df %>%
  filter(day %in% c('BL', '1WPI', '2WPI', '3WPI', '4WPI'),
         condition %in% c('wo', 'sham', 'STIM_40Hz', 'STIM_10KHz'),
         animal_id %in% c('M02', 'M03', 'M07', 'M09', 'M10', 'M13', 'M12')) %>%
  mutate(day = factor(day, levels = c('BL', '1WPI', '2WPI', '3WPI', '4WPI')))
         #condition = recode(condition, 'STIM_10KHz' = 'STIM', 'STIM_40Hz' = 'STIM'))

exclude_columns <- c('DLC_peak_times', 'DLC_start_times', 'DLC_end_times', 
                     'peaks_indice', 'start_of_rise_indices', 
                     'raw_force', 'smoothen_force', 'task_duration')

numerical_columns <- sapply(filtered_df, is.numeric)
columns_to_include <- names(numerical_columns[numerical_columns == TRUE])
columns_to_include <- setdiff(columns_to_include, exclude_columns)

# SEM function
sem <- function(x) {
  sd(x, na.rm = TRUE) / sqrt(length(na.omit(x)))
}

#colors for the conditions
condition_colors <- c('wo' = 'gray', 'sham' = 'gray', 'STIM_10KHz' = 'red', 'STIM_40KHz'= 'darkgray')

# directory to save plots
output_dir <- "/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/figure_nulle/fig_test_code_R"
dir.create(output_dir, showWarnings = FALSE)

# Plot with SEM as error bars
for (column in columns_to_include) {
  summary_df <- filtered_df %>%
    group_by(day, condition) %>%
    summarise(mean_value = mean(get(column), na.rm = TRUE),
              sem_value = sem(get(column)), .groups = 'drop')
  
  p <- ggplot(summary_df, aes(x = day, y = mean_value, fill = condition)) +
    geom_bar(stat = 'identity', position = position_dodge(width = 0.9)) +
    geom_errorbar(aes(ymin = mean_value - sem_value, ymax = mean_value + sem_value),
                  width = 0.2, position = position_dodge(width = 0.9)) +
    geom_point(data = filtered_df, aes(x = day, y = get(column), group = condition, fill = condition),
               position = position_dodge(width = 0.9),
               color = 'black', size = 2, alpha = 0.7, shape = 21, show.legend = FALSE) +
    labs(title = paste(column, 'for BL, wo, sham, STIM over 1WPI, 2WPI, 3WPI, 4WPI'),
         x = 'Day', y = column) +
    theme_minimal() +
    theme(legend.title = element_text(size = 10)) +
    scale_fill_manual(values = condition_colors) +
    guides(fill = guide_legend(override.aes = list(size = 3))) +
    theme(legend.position = "top")
  
  # Save the plot
ggsave(filename = paste0(output_dir, "/", column, "_bothhhh_plot_stim_param.pdf"), plot = p, width = 12, height = 8)
}
