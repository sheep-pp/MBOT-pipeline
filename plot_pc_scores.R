library(dplyr)
library(ggplot2)
library(tidyr)
library(scales)


# plot PCA, pc scores 
# uncomment and change directories to save the plots 

df <- read.csv('/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/normalized_mean_files.csv')

output_dir <- '/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/figure_nulle/plot_R'

# Filter 
filtered_df <- df %>%
  filter(day %in% c('BL', '1WPI', '2WPI', '3WPI', '4WPI'),
         condition %in% c('wo', 'sham', 'STIM_10KHz', 'STIM_40Hz'),
         animal_id %in% c('M02', 'M03', 'M07', 'M09', 'M10', 'M13', 'M12', 'M05', 'M08', 'M11', 'M14', 'M15', 'M01')) %>%
  mutate(day = factor(day, levels = c('BL', '1WPI', '2WPI', '3WPI', '4WPI'), ordered = TRUE),
         condition = recode(condition, 'STIM_10KHz' = 'STIM', 'STIM_40Hz' = 'STIM'))

exclude_columns <- c('DLC_peak_times', 'DLC_start_times', 'DLC_end_times', 
                     'peaks_indice', 'start_of_rise_indices', 
                     'raw_force', 'smoothen_force', 'task_duration')

numerical_columns <- names(select(filtered_df, where(is.numeric)))
columns_to_include <- setdiff(numerical_columns, exclude_columns)

# PCA
scaled_data <- scale(select(filtered_df, all_of(columns_to_include)))
pca <- prcomp(scaled_data, center = TRUE, scale. = TRUE)
explained_variance <- summary(pca)$importance[2, 1:2]

# create df with PC scores
pca_df <- as.data.frame(pca$x)
pca_df$Animal <- filtered_df$animal_id
pca_df$Day <- filtered_df$day
pca_df$Condition <- filtered_df$condition

# Calculate SEM
sem <- function(x) {
  sd(x, na.rm = TRUE) / sqrt(length(na.omit(x)))
}

# Compute means and SEMs for each group
group_means <- pca_df %>%
  group_by(Day, Condition) %>%
  summarise(PC1_mean = mean(PC1), PC2_mean = mean(PC2), 
            PC1_sem = sem(PC1), PC2_sem = sem(PC2), .groups = 'drop')

#condition_colors <- c('wo' = 'gray', 'sham' = 'gray', 'STIM_10KHz' = 'red', 'STIM_40Hz' = 'lightcoral')

condition_colors <- c('wo' = 'gray', 'sham' = 'gray', 'STIM' = 'red')
#pca with mean 
pca_plot <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Condition, shape = Day)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_point(data = group_means, aes(x = PC1_mean, y = PC2_mean), color = "black", size = 5, shape = 8) +
  geom_point(data = group_means, aes(x = PC1_mean, y = PC2_mean, color = Condition, shape = Day), size = 4, alpha = 1) +
  scale_color_manual(values = condition_colors) +
  scale_shape_manual(values = c('BL' = 4, '1WPI' = 16, '2WPI' = 17, '3WPI' = 15, '4WPI' = 18)) +
  labs(title = 'PCA (sham, STIM) all animals, 1,2,3,4WPI',
       x = paste0('PC1 (', round(explained_variance[1] * 100, 2), '% variance explained)'),
       y = paste0('PC2 (', round(explained_variance[2] * 100, 2), '% variance explained)')) +
  theme_minimal() +
  theme(legend.title = element_text(size = 10)) +
  theme(legend.position = "right") +
  theme(legend.title = element_text(size = 10), legend.text = element_text(size = 8))

print(pca_plot)

#  PC1 
pc1_plot <- ggplot(group_means, aes(x = Day, y = PC1_mean, fill = Condition)) +
  geom_bar(stat = 'identity', position = position_dodge()) +
  geom_errorbar(aes(ymin = PC1_mean - PC1_sem, ymax = PC1_mean + PC1_sem), 
                width = 0.2, position = position_dodge(0.9)) +
  geom_point(data = pca_df, aes(x = Day, y = PC1, fill = Condition),
             position = position_dodge(0.9),
             color = 'black', size = 2, alpha = 0.7, shape = 21, show.legend = FALSE) +
  labs(title = 'PC1 scores for BL, sham, and STIM over 1WPI, 2WPI, 3WPI, 4WPI',
       x = 'Day', y = 'PC1 score') +
  theme_minimal() +
  scale_fill_manual(values = condition_colors) +
  theme(legend.title = element_text(size = 10)) +
  guides(fill = guide_legend(override.aes = list(size = 3))) +
  theme(legend.position = "top")

print(pc1_plot)
# ggsave(filename = paste0(output_dir, "/all_animal_sham_both_stim_PC1_scores.pdf"), plot = pc1_plot, width = 12, height = 8)

#  PC2  
pc2_plot <- ggplot(group_means, aes(x = Day, y = PC2_mean, fill = Condition)) +
  geom_bar(stat = 'identity', position = position_dodge()) +
  geom_errorbar(aes(ymin = PC2_mean - PC2_sem, ymax = PC2_mean + PC2_sem), 
                width = 0.2, position = position_dodge(0.9)) +
  geom_point(data = pca_df, aes(x = Day, y = PC2, fill = Condition),
             position = position_dodge(0.9),
             color = 'black', size = 2, alpha = 0.7, shape = 21, show.legend = FALSE) +
  labs(title = 'PC2 scores for BL, sham, and STIM over 1WPI, 2WPI, 3WPI, 4WPI',
       x = 'Day', y = 'PC2 score') +
  theme_minimal() +
  scale_fill_manual(values = condition_colors) +
  theme(legend.title = element_text(size = 10)) +
  guides(fill = guide_legend(override.aes = list(size = 3))) +
  theme(legend.position = "top")

print(pc2_plot)
# ggsave(filename = paste0(output_dir, "/all_animal_sham_both_stim_PC2_scores.pdf"), plot = pc2_plot, width = 12, height = 8)
