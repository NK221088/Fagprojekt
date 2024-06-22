import csv
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

# Function to parse the accuracy dictionary string into an actual dictionary
def parse_accuracy_dict(accuracy_dict_str):
    """Parse the accuracy dictionary string into an actual dictionary."""
    # Define a safe evaluation context
    context = {"array": np.array, "int64": np.int64}
    parsed_dict = eval(accuracy_dict_str, {"__builtins__": None}, context)
    return parsed_dict

# Function to load the data from a CSV file
def load_data(file_path):
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        data = {row[0]: parse_accuracy_dict(row[1]) for row in reader}
    return data

# Load data from the CSV file
file_path = 'E_test_output_20240622_115428.csv'  # Replace with your file path
data = load_data(file_path)

# Extract models and predictions
models = list(data.keys())

predictions = {model: {fold: pred[3] for fold,pred in data[model].items()} for model in models}
True_labels = [predictions[models[0]][key][1] for key in predictions[models[0]].keys()]
True_labels = np.concatenate(True_labels).ravel() # Flattening list of true labels
predictions = {model: {fold: pred[0] for fold, pred in predictions[model].items()} for model in predictions.keys()}

True_labels = np.array([0., 1., 1., 0., 1., 0., 1., 1., 1., ... ])  # Correct labels

# Define the models to compare
model1_name = 'SVM'
model2_name = 'ANN'

# Initialize counts for McNemar's test
a = b = c = d = 0

# Iterate over each fold or sample to calculate counts
for fold in predictions[model1_name]:
    pred1 = predictions[model1_name]
    pred2 = predictions[model2_name]
    true = True_labels

    for i in range(len(true)):
        if pred1[i] == true[i] and pred2[i] == true[i]:
            d += 1
        elif pred1[i] == true[i] and pred2[i] != true[i]:
            b += 1
        elif pred1[i] != true[i] and pred2[i] == true[i]:
            c += 1
        elif pred1[i] != true[i] and pred2[i] != true[i]:
            a += 1

# Perform McNemar's test
statistic = ((b - c)**2) / (b + c)
p_value = 1 - chi2_contingency([[b, c], [a, d]])[1]

print(f"McNemar's test statistic: {statistic}")
print(f"P-value: {p_value}")