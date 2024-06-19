import pandas as pd


# How many of the top accuracies do we want?
X = 50


# Load the datasets
file_paths = [
    "./final_results/E_genList_output_20240618_211115.csv",
    "./final_results/E_genList_output_20240618_233126.csv",
    "./final_results/E_genList_output_20240619_003144.csv"
]

# Read the CSV files into DataFrames
dfs = [pd.read_csv(file) for file in file_paths]

# Combine the dataframes into one
combined_df = pd.concat(dfs)

# Convert Accuracy to numeric for sorting
combined_df['Accuracy'] = pd.to_numeric(combined_df['Accuracy'], errors='coerce')

# Function to get the top 20 accuracies for a given model
def get_top_accuracies(model_name, df):
    model_df = df[df['Model'] == model_name]
    top_accuracies = model_df.nlargest(X, 'Accuracy') 
    return top_accuracies

# Example usage for model 'ANN'
model_name = 'SVM'  # replace with your desired model name
top_accuracies = get_top_accuracies(model_name, combined_df)

# Display the results
print(top_accuracies)

# Optionally, save the results to a CSV file
top_accuracies.to_csv(f'top_{X}_accuracies_{model_name}.csv', index=False)
