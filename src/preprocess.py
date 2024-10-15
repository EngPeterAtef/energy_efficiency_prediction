# src/preprocess.py
"""
This file handles the preprocessing tasks like handling missing values,
encoding categorical variables, and feature scaling.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE
import numpy as np

def preprocess_data_regression(data):
    """Performs data preprocessing like missing value handling 
    and encoding."""
    # Remove any unnamed columns (might occur due to difference in pandas readers)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    # Feature selection
    data = data.drop('Y2', axis=1)
    X = data.drop('Y1', axis=1)
    y = data['Y1']
    return X, y
def preprocess_data_classification(data):
    """Performs data preprocessing like missing value handling 
    and encoding."""
    # Remove any unnamed columns (might occur due to difference in pandas readers)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # Handle missing values
    data.fillna(data.mean(), inplace=True)
    # Feature selection
    data = data.drop('Y1', axis=1)
    X = data.drop('Y2', axis=1)
    y = data['Y2']
    return X, y
def split_data(X, y):
    """Splits the data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
    test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_median = np.median(y_train)
    print("Median value of the target:", y_median)
    # Since we will treat this as a classification task, we will assume that
    # the load is "high" (y = True) if its compressive ratio is higher than the median
    # otherwise, it is assumed to be "low" (y = False)
    y_train = y_train > y_median
    y_test = y_test > y_median
    
    # Now ~50% of the samples should be considered "high" and the rest are considered "low"
    print(f"Percentage of 'high load' samples: {y_train.mean() * 100} %")
    # Also, lets standardize the data since it improves the training process
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean)/(1e-8 + X_std)
    X_test = (X_test - X_mean)/(1e-8 + X_std)
    return X_train, X_test, y_train, y_test