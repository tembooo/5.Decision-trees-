# 5.Decision-trees-
Decision trees : A decision tree is a decision support recursive partitioning structure that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility.
![image](https://github.com/user-attachments/assets/c8ff5469-72e9-4c29-85d1-a781452e20ee)

## 🌳 Building a Decision Tree for Classification

When building a decision tree for classification, it’s essential to determine the order in which features should be tested at each node.  
The selection of features can be guided by a measure of node purity. This helps decide how to split the dataset effectively at each point in the tree.

A common way to do this is by using a metric known as **information gain**, which evaluates how much uncertainty is reduced by a feature split.  
In essence, it measures how well a feature separates the data into distinct classes.

---

### 📘 Dataset Information

The dataset used here includes three features:
- author  
- thread  
- length  

The class label that needs to be predicted is:
- action

We need to determine which of these features provides the best split at the root node of the decision tree.

---

### 🔍 Feature Selection Process

Step-by-step outline:
1. Calculate the information gain for each feature to understand how useful each one is for classification.
2. The feature with the highest information gain will be selected to split the data at the root node.
3. This process is repeated recursively for each node in the tree.

---

### 📁 Dataset File

A sample dataset file is available for testing:
- **File**: `t116.csv`  
- **Source**: Poole, Mackworth (2010)

---

### 🛠 MATLAB Hint

To load and process the dataset in MATLAB:

```matlab
T = readtable('t116.csv');
T.Properties
```
```python
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
```
