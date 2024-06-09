import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Assuming the CSV file is saved as 'generated_csv_file.csv'
file_path = 'output_20240608_232043.csv'  # Replace with the actual path to your CSV file

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Convert the Accuracy column to numeric (in case it's read as a string)
df['Accuracy'] = pd.to_numeric(df['Accuracy'])

# Group by Evaluation and get the row with the maximum Accuracy for each fold
best_per_evaluation = df.loc[df.groupby('Evaluation')['Accuracy'].idxmax()]

# Parse the Parameter column to extract individual hyperparameter settings
hyperparameter_counts = Counter()
for param_string in best_per_evaluation['Parameter']:
    # Split the parameter string into individual settings
    params = re.split(r',\s*', param_string)
    # Update the counter with the current settings
    hyperparameter_counts.update(params)

# Convert the Counter to a DataFrame for easier plotting
hyperparameter_df = pd.DataFrame.from_dict(hyperparameter_counts, orient='index', columns=['Count']).reset_index()
hyperparameter_df.columns = ['Hyperparameter', 'Count']

# Save the result to a new CSV file if needed
output_file_path = 'best_hyperparameters_per_evaluation.csv'
best_per_evaluation.to_csv(output_file_path, index=False)

# Print the DataFrame with the best hyperparameters per evaluation
print(best_per_evaluation)

# Print the hyperparameter counts
print(hyperparameter_df)

# Plot the counts of each unique hyperparameter setting
plt.figure(figsize=(10, 6))
plt.barh(hyperparameter_df['Hyperparameter'], hyperparameter_df['Count'], color='skyblue')
plt.xlabel('Count')
plt.ylabel('Hyperparameter Setting')
plt.title('Frequency of Hyperparameter Settings Across Evaluations')
plt.gca().invert_yaxis()  # Invert y-axis to display the highest count at the top
plt.show()