#!/usr/bin/env python
# coding: utf-8

# ### LOAD LIBRARIES

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report


from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ### LOAD THE DATA

# In[ ]:


# Assuming the dataset is loaded into a DataFrame named data
heart_failure = pd.read_csv('heart_failure_clinical_records.csv')

heart_failure.info()


# In[ ]:


heart_failure.DEATH_EVENT.value_counts()


# ### SCALING THE DATA USING MINMAXSCALER

# In[ ]:


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


# In[ ]:


X_combination


# ### SPLIT THE DATASET
# 
# The X train will be used for training and validation in the `GridSearchCV` so as to determine the best parameters.
# 
# We will use the test set to validate our model with the best hyperparameter as provided by `GridSearchCV`.

# In[ ]:


# split the dataset
# Split the original training data into new training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combination, y, test_size=0.3, random_state=42)


# ### TRAIN THE MODEL

# In[ ]:


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


# x_train and y_train will be used in the cross-validation model
grid_search.fit(X_train, y_train)


# In[ ]:


# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters: ", best_params)


# ### USE THE BEST ESTIMATOR ON THE `X_test`

# In[ ]:


# Evaluate the best model on the test set
best_clf = grid_search.best_estimator_
y_pred_test_best = best_clf.predict(X_test)


# In[ ]:


# accuracy of the best estimator (model) on the test set
test_accuracy_best = accuracy_score(y_test, y_pred_test_best)
print("Best Model Test Accuracy:", test_accuracy_best)


# In[ ]:


# Classification report for the best estimator (model)
print("Classification Report for Best Model:")
print(classification_report(y_test, y_pred_test_best))


# ### TRAINING THE WHOLE DATASET AFTER HYPERTUNING

# In[ ]:


best_clf = MLPClassifier(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    learning_rate_init=best_params['learning_rate_init'],
    max_iter=best_params['max_iter'],
    batch_size=best_params['batch_size'],
    solver='sgd',
    learning_rate='constant',
    early_stopping=True,
    n_iter_no_change=20,
    verbose=0
)

best_clf.fit(X_combination, y)

# Save the final model
import joblib
joblib.dump(best_clf, 'final_best_model.pkl')
print("Final model saved to final_best_model.pkl")


# In[ ]:


from sklearn.model_selection import cross_val_score

# Perform cross-validation and print the mean accuracy
cv_scores = cross_val_score(best_clf, X_combination, y, cv=5, scoring='accuracy')
print("Cross-validation accuracy scores:", cv_scores)
print("Mean cross-validation accuracy:", cv_scores.mean())


# In[ ]:


Test_heart_failure = pd.read_csv('test_data2.csv')

Test_heart_failure.info()

# numerical columns
test_numerical_features = ['Age', 'creatinine_phosphokinase', 'ejection_fraction',
                      'platelets', 'serum_creatinine', 'serum_sodium', 'time']

# categorical columns
test_categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# Separate the columns to scale and the rest
X_numerical = Test_heart_failure[test_numerical_features]
X_categorical = Test_heart_failure[test_categorical_features]

# Apply one-hot encoding to categorical features
#X_categorial = pd.get_dummies(X_uncoded, drop_first=True)

#Apply MinMax scaling to the continuous features
scaler = MinMaxScaler()
X_numerical = scaler.fit_transform(X_numerical)
numerical_scaled = pd.DataFrame(X_numerical, columns=numerical_features)

#Combine scaled continuous features with one-hot encoded categorical features
X_combination_test = pd.concat([numerical_scaled, X_categorical], axis=1)


# In[ ]:


predictions = final_model.predict(X_combination_test)
print("Predictions on new data:", predictions)


# In[ ]:


Test_heart_failure = pd.read_csv('Testing Data.csv')

Test_heart_failure.info()

# numerical columns
test_numerical_features = ['Age', 'creatinine_phosphokinase', 'ejection_fraction',
                      'platelets', 'serum_creatinine', 'serum_sodium', 'time']

# categorical columns
test_categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']

# Separate the columns to scale and the rest
X_numerical = Test_heart_failure[test_numerical_features]
X_categorical = Test_heart_failure[test_categorical_features]

# Apply one-hot encoding to categorical features
#X_categorial = pd.get_dummies(X_uncoded, drop_first=True)

#Apply MinMax scaling to the continuous features
scaler = MinMaxScaler()
X_numerical = scaler.fit_transform(X_numerical)
numerical_scaled = pd.DataFrame(X_numerical, columns=numerical_features)

#Combine scaled continuous features with one-hot encoded categorical features
X_combination_test = pd.concat([numerical_scaled, X_categorical], axis=1)


# In[ ]:


# Load the final model from the file
final_model = joblib.load('final_best_model.pkl')


new_data = np.array([[0.872727, 0.992498, 0.879697, 0.878833, 0.099888, 0.67714, 0.298932, 0, 0, 0, 1, 1],
                    [0.4754545, 0.104210, 0.1686667, 0.3789314, 0.255618, 0.485714, 0.722420, 0, 0, 0, 1, 0]])
    


# Make predictions
predictions = final_model.predict(new_data)
print("Predictions on new data:", predictions)

