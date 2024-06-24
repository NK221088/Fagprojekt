import numpy as np
from scipy import stats

# Results from the two sets
results_set_1 = {
    'ANN': 0.74,
    'CNN': 0.66,
    'SVM': 0.64,
'Baseline': 0.66,
'Mean': 0.61
}

results_set_2 = {
    'ANN': 0.73,
    'CNN': 0.66,
    'SVM': 0.64,
'Baseline': 0.66,
'Mean': 0.62
}

results_set_3 = {
    'ANN': 0.76,
    'CNN': 0.67,
    'SVM': 0.64,
'Baseline': 0.66,
'Mean': 0.63
}

# Combine the results into a dictionary of lists
combined_results = {model: [results_set_1[model], results_set_2[model], results_set_3[model]] for model in results_set_1.keys()}

# Calculate the mean and confidence intervals
confidence_intervals = {}
for model, scores in combined_results.items():
    mean_score = np.mean(scores)
    sem = stats.sem(scores)  # Standard error of the mean
    ci = stats.t.interval(0.95, len(scores)-1, loc=mean_score, scale=sem)  # 95% confidence interval
    confidence_intervals[model] = {
        'mean': mean_score,
        'confidence_interval': ci
    }

# Print the results
for model, stats in confidence_intervals.items():
    print(f"{model} Classifier:")
    print(f"  Mean Score: {stats['mean']:.2f}")
    print(f"  95% Confidence Interval: ({stats['confidence_interval'][0]:.2f}, {stats['confidence_interval'][1]:.2f})\n")