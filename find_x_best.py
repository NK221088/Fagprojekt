import pandas as pd
from datetime import datetime
import os

# Load the datasets
file_paths = [
    "./Final_results/E_genList_output_20240621_042931.csv"
]

# Read the CSV files into DataFrames
dfs = [pd.read_csv(file) for file in file_paths]

# Combine the dataframes into one
combined_df = pd.concat(dfs)

# Convert Accuracy to numeric for sorting
combined_df['Accuracy'] = pd.to_numeric(combined_df['Accuracy'], errors='coerce')

# Function to get the best accuracy per fold for each model
def get_best_accuracy_per_fold(df):
    best_accuracies_list = []
    folds = df['Evaluation'].unique()
    
    for model_name in df['Model'].unique():
        for fold in folds:
            fold_df = df[(df['Model'] == model_name) & (df['Evaluation'] == fold)]
            if not fold_df.empty:
                best_accuracy = fold_df.loc[fold_df['Accuracy'].idxmax()]
                best_accuracies_list.append(best_accuracy)
    
    return pd.DataFrame(best_accuracies_list)

# Function to get the top 20 accuracies for a given model
def get_top_accuracies(model_name, df):
    if model_name == "ALL":
        return get_best_accuracy_per_fold(df)
    else:
        model_df = df[df['Model'] == model_name]
        top_accuracies = model_df.nlargest(20, 'Accuracy')
        return top_accuracies

# Example usage
model_name = 'ALL'  # replace with your desired model name or use 'ALL'
top_accuracies = get_top_accuracies(model_name, combined_df)

# Display the results
print(top_accuracies)

# Create directory if it doesn't exist
output_dir = os.path.join(os.getcwd(), 'parameter_table')
os.makedirs(output_dir, exist_ok=True)

# Optionally, save the results to a CSV file with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = os.path.join(output_dir, f'top_accuracies_{model_name}_{timestamp}.csv')
top_accuracies.to_csv(file_path, index=False)