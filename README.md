# Predicting-Heart-Failure
Including a code summary in your README is a good idea, especially if it helps users understand the core components and flow of your project. Here’s a revised version of the README with the code summary included:

---

# Heart Failure Prediction

This project uses a deep learning model, specifically the `MLPClassifier`, to predict the risk of heart failure among a diverse audience. The model is trained and evaluated on a dataset with various features related to heart health.

## Project Overview

The aim of this project is to build a predictive model that can estimate the likelihood of heart failure based on several input features. The key steps involved in the project include data scaling, train-test splitting, hyperparameter tuning using GridSearchCV, and model evaluation using Cross-Validation.

## Workflow

1. **Data Scaling**: The input data is scaled using `MinMaxScaler` to ensure that all features are within a similar range, which helps in speeding up the convergence of the neural network.

2. **Train-Test Split**: The dataset is split into training and testing sets to evaluate the performance of the model.

3. **Hyperparameter Tuning**: `GridSearchCV` is used to find the optimal hyperparameters for the `MLPClassifier`. This involves searching over a specified parameter grid and using cross-validation to select the best parameters.

4. **Model Evaluation**: The model is evaluated using `Cross-Val-Score` to ensure its robustness and reliability. This involves performing k-fold cross-validation to assess the model's performance on different subsets of the data.

## Code Summary

Here is a summary of the key code components used in the project:

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report


from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load data
heart_failure = pd.read_csv('heart_failure_clinical_records.csv')

heart_failure.info()

# Data Preprocessing
# Separate features and target
X = heart_failure.drop('DEATH_EVENT', axis=1)
y = heart_failure['DEATH_EVENT']

# numerical columns
numerical_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction',
                      'platelets', 'serum_creatinine', 'serum_sodium', 'time']

# categorical columns
categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# Separate the columns to scale and the rest
X_numerical = X[numerical_features]
X_categorical = X[categorical_features]

# Apply one-hot encoding to categorical features
#X_categorial = pd.get_dummies(X_uncoded, drop_first=True)

#Apply MinMax scaling to the continuous features
scaler = MinMaxScaler()
X_numerical = scaler.fit_transform(X_numerical)
numerical_scaled = pd.DataFrame(X_numerical, columns=numerical_features)

#Combine scaled continuous features with one-hot encoded categorical features
X_combination = pd.concat([numerical_scaled, X_categorical], axis=1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_combination, y, test_size=0.3, random_state=42)

# Define MLPClassifier
clf = MLPClassifier(
    solver='sgd',
    learning_rate='constant',
    early_stopping=True,
    n_iter_no_change=20,
    verbose=0
)

# Initialize the classifier with the best learning rate, epochs, and batch size
best_learning_rate = [0.001, 0.01, 0.1]
best_epochs = range(100, 600, 100)
best_batch_size = [i*32 for i in range(1, 5)]

# Define the parameter grid for hyperparameter tuning
param_grid = {'hidden_layer_sizes': [(100,100), (100,150), (100,200)],
              'learning_rate_init': best_learning_rate,
              'max_iter': best_epochs,
              "batch_size":best_batch_size}

# Perform grid search cross-validation for remaining hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')

# Best parameters found
best_params = grid_search.best_params_
print("Best Hyperparameters: ", best_params)

# Cross-Validation Score
cv_scores = cross_val_score(best_clf, X_combination, y, cv=5, scoring='accuracy')
print("Cross-validation accuracy scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())

```

## Results

The results of the project, including the best parameters found through GridSearchCV and the cross-validation scores, can be analyzed to understand the model's performance. Detailed results and model evaluation metrics will help in assessing the accuracy and reliability of the predictions.

## Conclusion

This project demonstrates the use of a deep learning model to predict heart failure risk, leveraging data scaling, train-test splitting, hyperparameter tuning, and cross-validation. The `MLPClassifier` provides a robust approach to handle complex patterns in the data, making it a suitable choice for this predictive modeling task.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.
