# src/utils.py
"""
This file contains utility functions, such as functions for visualizations or
logging.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(importances, feature_names):
    """Plots the feature importance of a model."""
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance")
    plt.show()
