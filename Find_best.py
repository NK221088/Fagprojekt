import pandas as pd
import os as os

# Load the CSV file into a DataFrame
file_path = "./E_genList_output_20240613_023704.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Group by Evaluation fold and find the max accuracy for each fold
max_accuracy_per_fold = df.loc[df.groupby('Evaluation')['Accuracy'].idxmax()]

# Print the results
print(max_accuracy_per_fold)

# Define the output file path
output_file_path = os.path.join(os.path.dirname(file_path), 'max_accuracy_per_fold.csv')

# Save the results to a new CSV file in the same folder
max_accuracy_per_fold.to_csv(output_file_path, index=False)

print(f"Results saved to {output_file_path}")