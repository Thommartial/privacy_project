"""
Data loading utilities.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_preprocess_data(data_path=None, max_samples=None):
    """Load and preprocess data."""
    if data_path is None:
        data_path = Path(__file__).parent.parent.parent / "data/processed"
    
    data_path = Path(data_path)
    
    if (data_path / "train.csv").exists():
        df = pd.read_csv(data_path / "train.csv")
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)
        return df
    else:
        # Return empty dataframe
        return pd.DataFrame({'text': [], 'label': []})

def split_dataset(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split dataset into train/val/test."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df
