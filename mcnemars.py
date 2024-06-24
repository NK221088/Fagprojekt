import csv
import os
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

# Function to save the results to a CSV file
def save_results_to_csv(results, folder_path, filename="mcnemar_results.csv"):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Model 1", "Model 2", "P-value", "Statistic"])
        for model_pair, result in results.items():
            model1, model2 = model_pair
            writer.writerow([model1, model2, result.pvalue, result.statistic])
    print(f"Results saved to {file_path}")

# Function to perform McNemar's test and save the results
def McNemar_results(file_paths, save_folder):
    _True_labels = []
    _model_predictions = {}

    for file_path in file_paths:
        data = load_data(file_path)

        # Extract models and predictions
        models = list(data.keys())

        predictions = {model: {fold: pred[3] for fold, pred in data[model].items()} for model in models}
        True_labels = [predictions[models[0]][key][1] for key in predictions[models[0]].keys()]
        True_labels = np.concatenate(True_labels).ravel()  # Flattening list of true labels
        predictions = {model: {fold: pred[0] for fold, pred in predictions[model].items()} for model in predictions.keys()}

        # Prepare predictions for each model
        model_predictions = {model: np.concatenate([predictions[model][fold] for fold in predictions[model].keys()]) for model in models}

        _True_labels.append(True_labels)

        # Add or append model predictions to _model_predictions
        for model in model_predictions:
            if model in _model_predictions:
                _model_predictions[model] = np.append(_model_predictions[model], model_predictions[model])
            else:
                _model_predictions[model] = model_predictions[model]
    model_predictions = _model_predictions
    True_labels = np.concatenate(_True_labels)
    
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

    # Save results to a CSV file
    save_results_to_csv(results, save_folder)

    # Optionally, print results
    for model_pair, result in results.items():
        model1, model2 = model_pair
        print(f"McNemar's test between {model1} and {model2}: p-value = {result.pvalue:.8f}, statistic = {result.statistic:.8f}")

# Example usage
file_path = ['Final_results_for_report\Tongue\E_test_output_20240623_180846.csv', 'Final_results_for_report\Tongue\E_test_output_20240624_003849.csv']  # Replace with your file path
save_folder = "Final_results_for_report\Tongue"
McNemar_results(file_path, save_folder)