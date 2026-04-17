"""
Data loading and preprocessing utilities for NCAA basketball data
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_ncaa_data(data_dir):
    """
    Load NCAA basketball data from CSV files.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        Dictionary containing loaded datasets
    """
    data_path = Path(data_dir)
    datasets = {}
    
    # Load available CSV files
    for csv_file in data_path.glob('*.csv'):
        try:
            datasets[csv_file.stem] = pd.read_csv(csv_file)
            print(f"Loaded {csv_file.stem}: {datasets[csv_file.stem].shape}")
        except Exception as e:
            print(f"Error loading {csv_file.stem}: {e}")
    
    return datasets


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X: Features array (n_samples, n_features)
        y: Labels array (n_samples,)
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    n_samples = X.shape[0]
    indices = np.random.permutation(n_samples)
    
    split_idx = int(n_samples * (1 - test_size))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize_data(X, mean=None, std=None):
    """
    Normalize data using z-score normalization.
    
    Args:
        X: Data array
        mean: Pre-computed mean (if None, computed from X)
        std: Pre-computed standard deviation (if None, computed from X)
        
    Returns:
        Normalized X, mean, std
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    
    # Avoid division by zero
    std = np.where(std == 0, 1, std)
    
    X_normalized = (X - mean) / std
    return X_normalized, mean, std


def batch_generator(X, y, batch_size=32, shuffle=True):
    """
    Generate batches of data for training.
    
    Args:
        X: Features array
        y: Labels array
        batch_size: Size of each batch
        shuffle: Whether to shuffle data between epochs
        
    Yields:
        (X_batch, y_batch) tuples
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_idx = indices[start_idx:end_idx]
        yield X[batch_idx], y[batch_idx]
