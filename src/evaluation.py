# src/evaluation.py
"""
This file evaluates the trained model and computes relevant performance
metrics.
"""
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error


def evaluate_model_regression(model, X_test, y_test):
    """Evaluates the model on the test data."""
    y_pred = model.predict(X_test)
    mean_squared_error_value = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mean_squared_error_value}")
    print(f"Evaluation Accuracy: {1 - mean_squared_error_value}")
    return 1 - mean_squared_error_value


def evaluate_model_classification(model, X_test, y_test):
    """Evaluates the model on the test data."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Evaluation Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
    return accuracy, report
