import pandas as pd
import numpy as np

# Load the dataset from the given file path
file_path = 'C:\\12.LUT\\00.Termic Cources\\2.pattern recognition\\jalase6\\Exersice\\t116.csv'
data = pd.read_csv(file_path)

# Function to calculate entropy for a given column of the dataset
def calculate_entropy(column):
    # Get unique elements and their corresponding counts
    unique_classes, class_counts = np.unique(column, return_counts=True)
    
    # Calculate the entropy using the probability of each class
    entropy_value = np.sum([(-class_counts[i] / np.sum(class_counts)) * np.log2(class_counts[i] / np.sum(class_counts)) 
                            for i in range(len(unique_classes))])
    return entropy_value

# Function to calculate the information gain for a given feature
def calculate_info_gain(dataset, feature_name, target_name="action"):
    # Calculate the entropy of the entire dataset (the target column)
    total_entropy = calculate_entropy(dataset[target_name])
    
    # Get the unique values and their counts for the feature
    feature_values, feature_counts = np.unique(dataset[feature_name], return_counts=True)
    
    # Calculate the weighted entropy for the feature
    weighted_entropy = np.sum([
        (feature_counts[i] / np.sum(feature_counts)) * 
        calculate_entropy(dataset[dataset[feature_name] == feature_values[i]][target_name])
        for i in range(len(feature_values))
    ])
    
    # Information gain is the original entropy minus the weighted entropy after the split
    info_gain_value = total_entropy - weighted_entropy
    return info_gain_value

# List of features to evaluate for information gain
features = ['author', 'thread', 'length']

# Calculate information gain for each feature
info_gain_results = {feature: calculate_info_gain(data, feature) for feature in features}

# Output the information gain for each feature
print("Information Gain for each feature:", info_gain_results)

# Determine the feature with the highest information gain
best_feature_to_split = max(info_gain_results, key=info_gain_results.get)
print(f"The best feature to split at the root node is: {best_feature_to_split}")