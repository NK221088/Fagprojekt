import csv
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

# Function to parse the accuracy dictionary string into an actual dictionary
def parse_accuracy_dict(accuracy_dict_str):
    """Parse the accuracy dictionary string into an actual dictionary."""
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

predictions = {model: {fold: pred[3] for fold, pred in data[model].items()} for model in models}
True_labels = [predictions[models[0]][key][1] for key in predictions[models[0]].keys()]
True_labels = np.concatenate(True_labels).ravel()  # Flattening list of true labels
predictions = {model: {fold: pred[0] for fold, pred in predictions[model].items()} for model in predictions.keys()}

# Prepare predictions for each model
model_predictions = {model: np.concatenate([predictions[model][fold] for fold in predictions[model].keys()]) for model in models}

# McNemar's test for each pair of models
def mcnemar_test(model1_preds, model2_preds, true_labels):
    # Contingency table
    b = np.sum((model1_preds == true_labels) & (model2_preds != true_labels))
    c = np.sum((model1_preds != true_labels) & (model2_preds == true_labels))
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return result

# Perform McNemar's test between all pairs of models
results = {}
for i, model1 in enumerate(models):
    for j, model2 in enumerate(models):
        if i < j:
            result = mcnemar_test(model_predictions[model1], model_predictions[model2], True_labels)
            results[(model1, model2)] = result

# Print results
for model_pair, result in results.items():
    model1, model2 = model_pair
    print(f"McNemar's test between {model1} and {model2}: p-value = {result.pvalue:.4f}, statistic = {result.statistic:.4f}")