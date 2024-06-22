import pandas as pd
import os

# List of correct files
file_paths = [
    './parameter_table/top_accuracies_ALL_20240621_131641.csv',
    './parameter_table/top_accuracies_ALL_20240621_131658.csv',
    './parameter_table/top_accuracies_ALL_20240621_131702.csv',
    './parameter_table/top_accuracies_ALL_20240621_131709.csv',
    './parameter_table/top_accuracies_ALL_20240621_131718.csv'
]

# Initialize an empty list to store dataframes
dfs = []

# Load each CSV file into a dataframe and store it in the list
for file_path in file_paths:
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all dataframes into a single dataframe
combined_df = pd.concat(dfs, ignore_index=True)
combined_df = df_filtered = combined_df[~combined_df.apply(lambda row: row.astype(str).str.contains('Baseline').any(), axis=1)]
combined_df.reset_index(drop=True, inplace=True)
param_dict = {}

for i in range(combined_df.shape[0]):
    parameter_list = [param.strip() for param in  combined_df["Parameter"][i].split(',')]
    if "coef0=0" in parameter_list:
        parameter_list.remove('coef0=0')
    # print("lol")
    for i in parameter_list:
        if tuple(i.split('=')) in param_dict.keys():
            param_dict[tuple(i.split('='))] += 1
        else:
            param_dict[tuple(i.split('='))] = 1


print(param_dict)
