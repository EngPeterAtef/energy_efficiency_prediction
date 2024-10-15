# src/main.py
"""
This file is the entry point for the project and ties together all the other
components.
"""
from data_loader import *
from preprocess import *
from model import create_model, train_model, save_model
from evaluation import *
import config
import mlflow

def main():
    mlflow.start_run()
    # Load and explore data
    data = load_excel_data()
    explore_data(data)
    # Preprocess the data
    if config.MODEL_TYPE == "linear_regression":
        X, y = preprocess_data_regression(data)
    else:
        X, y = preprocess_data_classification(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    # Create, train, and evaluate the model
    model = create_model()
    # Log hyperparameters
    mlflow.log_param("model_type", config.MODEL_TYPE)
    mlflow.log_param("test_size", config.TEST_SIZE)

    trained_model = train_model(model, X_train, y_train)
    # Evaluate the model
    if config.MODEL_TYPE == "linear_regression":
        accuracy = evaluate_model_regression(trained_model, X_test, y_test)
    else:
        res = evaluate_model_classification(trained_model, X_test, y_test)
        accuracy = res[0]
    # After model evaluation
    mlflow.log_metric("accuracy", accuracy)
    mlflow.end_run()
    # Save the model
    save_model(trained_model)
    
    
if __name__ == "__main__":
    main()
