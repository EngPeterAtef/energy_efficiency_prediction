# src/data_loader.py
"""
This file is responsible for loading the dataset and performing initial
exploration.
"""
import pandas as pd
from config import DATA_PATH
def load_csv_data():
    """Loads the dataset from the specified path."""
    data = pd.read_csv(DATA_PATH)
    return data
def explore_data(data):
    """Returns basic information and description of the 
    dataset."""
    print(data.info())
    print(data.describe())
    print(data.head())
    
def load_excel_data():
    """Loads the dataset from the specified path."""
    df = pd.read_excel(DATA_PATH, engine = 'openpyxl')
    return df
