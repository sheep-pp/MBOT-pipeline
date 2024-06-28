import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

csv_file = '/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/normalized_mean_files.csv'
df = pd.read_csv(csv_file)

'''pca only sham accros all days, its loading factor, pc score bar plot
'''


# Filter : in here you choose the weeks to plot, which condition (stim, bl etc) and which animal. 
filtered_df = df[(df['day'].isin(['BL', '1WPI', '2WPI', '3WPI',  '4WPI', '5WPI', '6WPI', '7WPI', '8WPI'])) & 
                 (df['condition'].isin(['sham', 'wo', 'STIM_10KHz']))]
                 #&(df['animal_id'].isin([''M02', 'M03', 'M07', 'M09', 'M10', 'M13', 'M12']))]


filtered_df['day'] = pd.Categorical(filtered_df['day'], categories=['BL', '1WPI', '2WPI', '3WPI', '4WPI', '5WPI', '6WPI', '7WPI', '8WPI'], ordered=True)

# Exclude  columns : the one you don't want to have in the loading factor, pca etc
exclude_columns = [
    'DLC_peak_times', 'DLC_start_times', 'DLC_end_times', 
    'peaks_indice', 'start_of_rise_indices', 
    'raw_force', 'smoothen_force', 'task_duration'
]

numerical_columns = filtered_df.select_dtypes(include=[float, int]).columns
columns_to_include = [col for col in numerical_columns if col not in exclude_columns]
numerical_df = filtered_df[columns_to_include]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_df)

#  PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_

# df with pc
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Animal'] = filtered_df['animal_id'].values
pca_df['Day'] = filtered_df['day'].values
pca_df['Condition'] = filtered_df['condition'].values

# Plot PCA 
plt.figure(figsize=(10, 6))
colors = plt.cm.get_cmap('tab10', len(pca_df['Day'].unique()))

for i, day in enumerate(pca_df['Day'].unique()):
    day_data = pca_df[pca_df['Day'] == day]
    plt.scatter(day_data['PC1'], day_data['PC2'], alpha=0.7, label=day, color=colors(i))
    day_mean = day_data[['PC1', 'PC2']].mean()
    plt.scatter(day_mean['PC1'], day_mean['PC2'], color=colors(i), edgecolors='black', s=100, zorder=5)


# comment out to have the label on each point 

    #for _, row in day_data.iterrows():
        #plt.text(row['PC1'], row['PC2'], f"{row['Animal']}", fontsize=8, ha='right')

plt.title('PCA (BL, 1WPI, 2WPI, 3WPI, 4WPI) for sham')
plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}% variance explained)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}% variance explained)')
plt.legend(title='Day')
plt.grid(True)
plt.show()

# Plot loading factors 

# R= 25, G=211, B=197   [ (25/255, 211/255, 197/255),
cmap = LinearSegmentedColormap.from_list('impulse_Red',['white', (192/255, 226/255, 228/255), (25/255, 211/255, 197/255)])
loadings = pca.components_.T
loading_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=numerical_df.columns)

plt.figure(figsize=(12, 8))
sns.heatmap(loading_df, cmap=cmap)
plt.title('Loading factors')
plt.show()

# Bar plot of PC scores
pca_df['Day'] = pd.Categorical(pca_df['Day'], categories=['BL', '1WPI', '2WPI', '3WPI', '4WPI', '5WPI', '6WPI', '7WPI', '8WPI'], ordered=True)
bar_df = pca_df[pca_df['Condition'].isin(['sham', 'wo', 'STIM_10KHz'])]

plt.figure(figsize=(12, 8))
sns.barplot(data=bar_df, x='Day', y='PC1', hue='Condition', ci='sd')
sns.stripplot(data=bar_df, x='Day', y='PC1', hue='Condition', dodge=True, marker='o', alpha=0.7, color='black')
plt.title('PC1 scores for BL and sham over 1WPI, 2WPI, 3WPI, 4WPI')
plt.xlabel('Day')
plt.ylabel('PC1 score')
plt.legend(title='Condition')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(data=bar_df, x='Day', y='PC2', hue='Condition', ci='sd')
sns.stripplot(data=bar_df, x='Day', y='PC2', hue='Condition', dodge=True, marker='o', alpha=0.7, color='black')
plt.title('PC2 scores for BL and sham over 1WPI, 2WPI, 3WPI, 4WPI')
plt.xlabel('Day')
plt.ylabel('PC2 score')
plt.legend(title='Condition')
plt.show()

###################### indiv_param  ##############################@
'''df = pd.read_csv(csv_file)

# Filter 
filtered_df = df[(df['day'].isin(['BL', '1WPI', '2WPI', '3WPI', '4WPI'])) & (df['condition'].isin(['sham', 'wo', 'STIM_10KHz', 'STIM_40Hz']))]

filtered_df['day'] = pd.Categorical(filtered_df['day'], categories=['BL', '1WPI', '2WPI', '3WPI', '4WPI'], ordered=True)
 
exclude_columns = [
    'DLC_peak_times', 'DLC_start_times', 'DLC_end_times', 
    'peaks_indice', 'start_of_rise_indices', 
    'raw_force', 'smoothen_force', 'task_duration'
]

numerical_columns = filtered_df.select_dtypes(include=[float, int]).columns
columns_to_include = [col for col in numerical_columns if col not in exclude_columns]

# Plot individual parameters
for column in columns_to_include:
    plt.figure(figsize=(12, 8))
    sns.barplot(data=filtered_df, x='day', y=column, hue='condition', ci=95)  # Use SEM by setting ci to 95%
    sns.stripplot(data=filtered_df, x='day', y=column, hue='condition', dodge=True, marker='o', alpha=0.7, color='black')
    plt.title(f'{column} Scores for BL and sham over 1WPI, 2WPI, 3WPI, 4WPI')
    plt.xlabel('Day')
    plt.ylabel(column)
    plt.legend(title='Condition')
    plt.show()'''




# comment out this section to plot pca with stim
'''csv_file = '/Users/rimsadry/Documents/stage_EPFL_NeuroRestore/M-BOT/normalized_mean_files.csv'

df = pd.read_csv(csv_file)

# comment out to gather both stim in a single variable
#df['condition'] = df['condition'].replace(['STIM_10KHz', 'STIM_40Hz'], 'STIM')

# Filter
filtered_df = df[(df['day'].isin(['BL', '1WPI', '2WPI', '3WPI', '4WPI'])) & 
                 (df['condition'].isin(['wo','sham', 'STIM_10KHz', 'STIM_40Hz'])) &
                 (df['animal_id'].isin(['M02', 'M03', 'M07', 'M09', 'M10', 'M13', 'M12']))]

filtered_df['day'] = pd.Categorical(filtered_df['day'], categories=['BL', '1WPI', '2WPI', '3WPI', '4WPI'], ordered=True)

exclude_columns = [
    'DLC_peak_times', 'DLC_start_times', 'DLC_end_times', 
    'peaks_indice', 'start_of_rise_indices', 
    'raw_force', 'smoothen_force', 'task_duration'
]

numerical_columns = filtered_df.select_dtypes(include=[float, int]).columns
columns_to_include = [col for col in numerical_columns if col not in exclude_columns]


scaler = StandardScaler()
scaled_data = scaler.fit_transform(filtered_df[columns_to_include])

#  PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_

# df with pc 
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Animal'] = filtered_df['animal_id'].values
pca_df['Day'] = filtered_df['day'].values
pca_df['Condition'] = filtered_df['condition'].values

#plot pca
plt.figure(figsize=(10, 6))
colors = plt.cm.get_cmap('tab10', len(pca_df['Day'].unique()))

for i, day in enumerate(pca_df['Day'].unique()):
    day_data = pca_df[pca_df['Day'] == day]
    plt.scatter(day_data['PC1'], day_data['PC2'], alpha=0.7, label=day, color=colors(i))
    day_mean = day_data[['PC1', 'PC2']].mean()
    plt.scatter(day_mean['PC1'], day_mean['PC2'], color=colors(i), edgecolors='black', s=100, zorder=5)

plt.title('PCA for BL, wo, sham, and STIM (combined) across all days')
plt.xlabel(f'PC1 ({explained_variance[0]*100:.2f}% variance explained)')
plt.ylabel(f'PC2 ({explained_variance[1]*100:.2f}% variance explained)')
plt.legend(title='Day')
plt.grid(True)
plt.show()

# Plot loading factors 
loadings = pca.components_.T
loading_df = pd.DataFrame(loadings, columns=['PC1', 'PC2'], index=columns_to_include)

plt.figure(figsize=(12, 8))
sns.heatmap(loading_df, cmap='coolwarm')
plt.title('Loading factors')
plt.show()

# Bar plot of PC scores 
pca_df['Day'] = pd.Categorical(pca_df['Day'], categories=['BL', '1WPI', '2WPI', '3WPI', '4WPI'], ordered=True)
bar_df = pca_df[pca_df['Condition'].isin(['wo', 'sham', 'STIM_10KHz', 'STIM_40Hz'])]

plt.figure(figsize=(12, 8))
sns.barplot(data=bar_df, x='Day', y='PC1', hue='Condition', ci=95)
sns.stripplot(data=bar_df, x='Day', y='PC1', hue='Condition', dodge=True, marker='o', alpha=0.7, color='black')
plt.title('PC1 scores for BL, sham, and STIM over 1WPI, 2WPI, 3WPI, 4WPI')
plt.xlabel('Day')
plt.ylabel('PC1 score')
plt.legend(title='Condition')
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(data=bar_df, x='Day', y='PC2', hue='Condition', ci=95)
sns.stripplot(data=bar_df, x='Day', y='PC2', hue='Condition', dodge=True, marker='o', alpha=0.7, color='black')
plt.title('PC2 scores for BL, sham, and STIM over 1WPI, 2WPI, 3WPI, 4WPI')
plt.xlabel('Day')
plt.ylabel('PC2 score')
plt.legend(title='Condition')
plt.show()'''