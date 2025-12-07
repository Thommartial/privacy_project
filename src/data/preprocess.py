#!/usr/bin/env python3
"""
Data preprocessing and splitting script for PII-Masking-43K dataset.
Correctly maps: 'Filled Template' -> text, 'Tokens' -> labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import ast

# Configuration
TOTAL_SAMPLES = 5000
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
DATA_PATH = Path("/home/thom/Desktop/dpjax/data/raw/PII-Masking-43K.csv")

def load_dataset():
    """Load CSV with correct column mapping"""
    print(f"Loading dataset from: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")
    
    # Load with error skipping for malformed rows
    df = pd.read_csv(DATA_PATH, on_bad_lines='skip')
    print(f"✅ Loaded {len(df)} samples")
    
    return df

def preprocess_data(df, n_samples=TOTAL_SAMPLES):
    """Clean and prepare data with correct column mapping"""
    print(f"\nPreprocessing {n_samples} samples...")
    print(f"Original columns: {list(df.columns)}")
    
    # Map columns correctly based on the data structure
    # 'Filled Template' contains the actual text with PII
    # 'Tokens' contains BIO labels
    if 'Filled Template' in df.columns:
        df = df.rename(columns={'Filled Template': 'text'})
        print("✅ Using 'Filled Template' as text")
    else:
        raise ValueError("'Filled Template' column not found")
    
    if 'Tokens' in df.columns:
        df = df.rename(columns={'Tokens': 'labels'})
        print("✅ Using 'Tokens' as labels")
    else:
        raise ValueError("'Tokens' column not found")
    
    # Keep only needed columns
    df = df[['text', 'labels']]
    
    # Sample data
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=SEED, replace=False)
    else:
        print(f"⚠️  Dataset has only {len(df)} samples, using all")
        n_samples = len(df)
    
    # Clean text - remove extra whitespace
    df['text'] = df['text'].astype(str).str.strip()
    
    # Convert labels from string representation to list
    def parse_labels(label_str):
        """Convert string like "['O', 'B-NAME', ...]" to list"""
        if pd.isna(label_str) or label_str == '':
            return []
        
        try:
            # Remove quotes and parse
            if isinstance(label_str, str):
                # Handle the string representation of list
                return ast.literal_eval(label_str)
            elif isinstance(label_str, list):
                return label_str
            else:
                return []
        except (ValueError, SyntaxError) as e:
            print(f"⚠️  Could not parse labels: {e}")
            return []
    
    df['labels'] = df['labels'].apply(parse_labels)
    
    # Verify conversion
    sample_labels = df['labels'].iloc[0] if len(df) > 0 else []
    print(f"✅ Labels converted. Sample label type: {type(sample_labels)}, length: {len(sample_labels)}")
    
    print(f"✅ Preprocessed {len(df)} samples")
    return df

def split_data(df):
    """Split data into train/val/test"""
    print(f"\nSplitting data: {TRAIN_RATIO*100:.0f}% train, "
          f"{VAL_RATIO*100:.0f}% val, {TEST_RATIO*100:.0f}% test")
    
    # Shuffle
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # Calculate split indices
    n_train = int(len(df) * TRAIN_RATIO)
    n_val = int(len(df) * VAL_RATIO)
    
    # Split
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    print(f"✅ Train: {len(train_df)} samples")
    print(f"✅ Val: {len(val_df)} samples")
    print(f"✅ Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, output_dir=Path("data/processed")):
    """Save splits and statistics"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV files
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    # Count PII entities (non-O tags)
    def count_pii_entities(labels):
        if not isinstance(labels, list):
            return 0
        # Count labels that are not 'O'
        return sum(1 for label in labels if label != 'O')
    
    train_pii = sum(train_df['labels'].apply(count_pii_entities))
    val_pii = sum(val_df['labels'].apply(count_pii_entities))
    test_pii = sum(test_df['labels'].apply(count_pii_entities))
    
    # Save statistics
    stats = {
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "split_ratio": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO
        },
        "pii_statistics": {
            "train_pii_count": int(train_pii),
            "val_pii_count": int(val_pii),
            "test_pii_count": int(test_pii),
            "train_pii_per_sample": float(train_pii / len(train_df)) if len(train_df) > 0 else 0,
            "val_pii_per_sample": float(val_pii / len(val_df)) if len(val_df) > 0 else 0,
            "test_pii_per_sample": float(test_pii / len(test_df)) if len(test_df) > 0 else 0
        }
    }
    
    with open(output_dir / "split_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Saved splits to {output_dir}/")
    return stats

def analyze_dataset(df):
    """Basic dataset analysis"""
    print("\n📊 Dataset Analysis:")
    print("-" * 40)
    
    # Text length statistics
    text_lengths = df['text'].str.len()
    print(f"Text length - Avg: {text_lengths.mean():.1f} chars")
    print(f"Text length - Min: {text_lengths.min()} chars")
    print(f"Text length - Max: {text_lengths.max()} chars")
    
    # Token count (approximate by splitting on spaces)
    token_counts = df['text'].str.split().str.len()
    print(f"Tokens - Avg: {token_counts.mean():.1f} words")
    
    # Count PII entities (non-O tags)
    def count_pii(labels):
        if not isinstance(labels, list):
            return 0
        return sum(1 for label in labels if label != 'O')
    
    df['pii_count'] = df['labels'].apply(count_pii)
    total_pii = df['pii_count'].sum()
    pii_per_sample = total_pii / len(df) if len(df) > 0 else 0
    
    print(f"\nPII Statistics:")
    print(f"Total PII entities: {int(total_pii)}")
    print(f"PII per sample: {pii_per_sample:.2f}")
    print(f"Samples with PII: {(df['pii_count'] > 0).sum()} ({(df['pii_count'] > 0).sum()/len(df)*100:.1f}%)")
    
    # Analyze PII types distribution
    pii_types = {}
    for labels in df['labels']:
        if isinstance(labels, list):
            for label in labels:
                if label != 'O' and '-' in label:
                    pii_type = label.split('-')[1]  # Extract type from B-TYPE or I-TYPE
                    pii_types[pii_type] = pii_types.get(pii_type, 0) + 1
    
    print(f"\nPII Type Distribution:")
    for pii_type, count in sorted(pii_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_pii * 100 if total_pii > 0 else 0
        print(f"  {pii_type}: {count} ({percentage:.1f}%)")
    
    return {
        "avg_text_length": float(text_lengths.mean()),
        "avg_token_count": float(token_counts.mean()),
        "total_pii": int(total_pii),
        "pii_per_sample": float(pii_per_sample),
        "samples_with_pii": int((df['pii_count'] > 0).sum()),
        "pii_types": pii_types
    }

def main():
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("📊 DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Input: {DATA_PATH}")
    print(f"Samples: {TOTAL_SAMPLES}")
    print(f"Split: {TRAIN_RATIO*100:.0f}%/{VAL_RATIO*100:.0f}%/{TEST_RATIO*100:.0f}%")
    print("=" * 60)
    
    # Load data
    df = load_dataset()
    
    # Preprocess
    df = preprocess_data(df, TOTAL_SAMPLES)
    
    # Analyze
    analysis = analyze_dataset(df)
    
    # Split
    train_df, val_df, test_df = split_data(df)
    
    # Save
    stats = save_splits(train_df, val_df, test_df)
    
    # Combine all stats
    final_stats = {**stats, "analysis": analysis}
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Train: {stats['train_samples']} samples, {stats['pii_statistics']['train_pii_count']} PII entities")
    print(f"Val: {stats['val_samples']} samples, {stats['pii_statistics']['val_pii_count']} PII entities")
    print(f"Test: {stats['test_samples']} samples, {stats['pii_statistics']['test_pii_count']} PII entities")
    
    # Save full analysis
    with open("data/processed/data_analysis.json", "w") as f:
        json.dump(final_stats, f, indent=2)
    
    print(f"\n📁 Output saved to: data/processed/")
    
    return final_stats

if __name__ == "__main__":
    main()
