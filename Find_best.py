import pandas as pd
import os
from seed import set_seeds
set_seeds()

# Load the CSV file into a DataFrame
file_path = "./E_genList_output_20240617_114700.csv"  # Update with your actual file path
df = pd.read_csv(file_path)

# Group by Model and Evaluation fold and find the max accuracy for each combination
max_accuracy_per_model_fold = df.loc[df.groupby(['Model', 'Evaluation'])['Accuracy'].idxmax()]

# Print the results
print(max_accuracy_per_model_fold)

# Define the output file path
output_file_path = os.path.join(os.path.dirname(file_path), 'max_accuracy_per_model_fold.csv')

# Save the results to a new CSV file in the same folder
max_accuracy_per_model_fold.to_csv(output_file_path, index=False)

print(f"Results saved to {output_file_path}")
