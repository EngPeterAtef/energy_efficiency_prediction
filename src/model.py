# src/model.py
"""
This file builds the machine learning models, trains them, and saves the
best-performing models.
"""
from sklearn.ensemble import RandomForestClassifier
import joblib
from config import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

def create_model():
    """Creates a machine learning model based on the 
    configuration."""
    if MODEL_TYPE == 'random_forest':
        model = RandomForestClassifier(
        n_estimators=HYPERPARAMS['random_forest']
        ['n_estimators'], 
        max_depth=HYPERPARAMS['random_forest']['max_depth'], 
        random_state=RANDOM_STATE
        )
    elif MODEL_TYPE == 'linear_regression':
        model = LinearRegression()
    elif MODEL_TYPE == 'logistic_regression':
        model = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=HYPERPARAMS['logistic_regression']['max_iter']
        )
    return model

def train_model(model, X_train, y_train):
    """Trains the machine learning model."""
    model.fit(X_train, y_train)
    return model

def save_model(model):
    """Saves the trained model to a file."""
    joblib.dump(model, MODEL_SAVE_PATH)
