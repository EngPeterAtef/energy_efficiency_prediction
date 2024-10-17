# src/config.py
"""_summary_
    This file holds configuration settings and hyperparameters, making it easy to
update parameters across the project.
"""
# Configuration for the AI project
# Paths
DATA_PATH = "./data/Energy_Efficiency.xlsx"
# Hyperparameters for model training
RANDOM_STATE = 42
TEST_SIZE = 0.2
# MODEL_TYPE = "logistic_regression"
MODEL_TYPE = "random_forest"
# MODEL_TYPE = "linear_regression"
MODEL_SAVE_PATH = f"./models/{MODEL_TYPE}_model.pkl"
# Model-specific hyperparameters
HYPERPARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
    },
    "xgboost": {
        "learning_rate": 0.01,
        "n_estimators": 200,
        "max_depth": 5,
    },
    "logistic_regression": {
        "penalty": "none",
        "max_iter": 1000,
    },
}
