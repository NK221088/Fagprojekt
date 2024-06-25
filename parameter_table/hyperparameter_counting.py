import pandas as pd

# List of file paths
tapping_file_paths = [
    './parameter_table/top_accuracies_ALL_20240624_225936.csv',
    './parameter_table/top_accuracies_ALL_20240624_225936.csv',
    './parameter_table/top_accuracies_ALL_20240624_230035.csv',
]

# Initialize an empty list to store dataframes
tapping_dfs = []

# Load each CSV file into a dataframe and store it in the list
for file_path in tapping_file_paths:
    df = pd.read_csv(file_path)
    tapping_dfs.append(df)

# Concatenate all dataframes into a single dataframe
tapping_combined_df = pd.concat(tapping_dfs, ignore_index=True)
tapping_combined_df = tapping_combined_df[~tapping_combined_df.apply(lambda row: row.astype(str).str.contains('Baseline').any(), axis=1)]
tapping_combined_df.reset_index(drop=True, inplace=True)

# Dictionary to hold the count of parameter combinations
tapping_param_combination_dict = {}

for i in range(tapping_combined_df.shape[0]):
    parameter_list = [param.strip() for param in tapping_combined_df["Parameter"][i].split(',')]
    if "coef0=0" in parameter_list:
        parameter_list.remove('coef0=0')
    param_tuple = tuple(sorted(parameter_list))
    if param_tuple in tapping_param_combination_dict:
        tapping_param_combination_dict[param_tuple] += 1
    else:
        tapping_param_combination_dict[param_tuple] = 1

# Convert the dictionary to a dataframe for better visualization
tapping_param_combination_df = pd.DataFrame(list(tapping_param_combination_dict.items()), columns=['Parameter Combination', 'Count'])

# Sorting the dataframe by count
tapping_param_combination_df = tapping_param_combination_df.sort_values(by='Count', ascending=False).reset_index(drop=True)

# Save the parameter combination counts to a CSV file
tapping_param_combination_df.to_csv('tapping_parameter_combinations_count.csv', index=False)

# Initialize empty dictionaries for individual parameter counts
tapping_param_dict = {}

for i in range(tapping_combined_df.shape[0]):
    parameter_list = [param.strip() for param in tapping_combined_df["Parameter"][i].split(',')]
    if "coef0=0" in parameter_list:
        parameter_list.remove('coef0=0')
    for param in parameter_list:
        key_value_pair = tuple(param.split('='))
        if key_value_pair in tapping_param_dict:
            tapping_param_dict[key_value_pair] += 1
        else:
            tapping_param_dict[key_value_pair] = 1

# Define keys for ANN, CNN, and SVM
ann_keys = ['neuron1', 'neuron2', 'learning_rate', 'model', 'layers', 'use_svm', 'use_transfer_learning']
cnn_keys = ['batch_size', 'number_of_layers', 'base_learning_rate']
svm_keys = ['gamma', 'coef0', 'C', 'kernel', 'degree']

# Initialize empty dictionaries for each type of model
tapping_ann_dict = {}
tapping_cnn_dict = {}
tapping_svm_dict = {}

# Populate dictionaries based on keywords
for key, value in tapping_param_dict.items():
    if key[0] in ann_keys:
        tapping_ann_dict[key] = value
    elif key[0] in cnn_keys:
        tapping_cnn_dict[key] = value
    elif key[0] in svm_keys:
        tapping_svm_dict[key] = value

# Convert individual parameter dictionaries to dataframes
tapping_ann_df = pd.DataFrame(list(tapping_ann_dict.items()), columns=['Parameter', 'Count'])
tapping_cnn_df = pd.DataFrame(list(tapping_cnn_dict.items()), columns=['Parameter', 'Count'])
tapping_svm_df = pd.DataFrame(list(tapping_svm_dict.items()), columns=['Parameter', 'Count'])

# Save individual parameter dataframes to CSV files
tapping_ann_df.to_csv('tapping_ann_parameters_count.csv', index=False)
tapping_cnn_df.to_csv('tapping_cnn_parameters_count.csv', index=False)
tapping_svm_df.to_csv('tapping_svm_parameters_count.csv', index=False)