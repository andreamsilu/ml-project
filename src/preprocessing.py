import pandas as pd
import numpy as np

def load_data(file_path):
    """Load data from a CSV file."""
    data = pd.read_csv(file_path, header=None)
    features = data.iloc[:, :-1].values  # All but the last column
    labels = data.iloc[:, -1].values  # Last column
    return features, labels

def preprocess_data(train_data, test_data):
    """Normalize features to the range [0, 1]."""
    train_data = train_data / 16.0  # Assuming pixel values are between 0-16
    test_data = test_data / 16.0
    return train_data, test_data
