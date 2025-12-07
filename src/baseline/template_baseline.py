#!/usr/bin/env python3
"""
Simple baseline using template information.
Since templates contain [FULLNAME_1], [CITY_1] markers,
we can perfectly identify where PII should be.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
from sklearn.metrics import precision_recall_fscore_support

def load_data():
    """Load preprocessed splits"""
    print("📂 Loading preprocessed data...")
    
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Convert labels from string to list
    def parse_labels(labels_str):
        if isinstance(labels_str, str):
            return eval(labels_str)  # Convert "[...]" to list
        return labels_str
    
    for df in [train_df, val_df, test_df]:
        df['labels'] = df['labels'].apply(parse_labels)
    
    print(f"✅ Loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

def create_baseline_predictions(text, template):
    """
    Create baseline predictions using template markers.
    Template example: "Discuss [FULLNAME_1] who lives in [CITY_1]"
    Filled example:   "Discuss John Doe who lives in Paris"
    """
    # Find all template markers like [FULLNAME_1], [CITY_2], etc.
    markers = re.findall(r'\[([A-Z]+_\d+)\]', template)
    
    # For now, return dummy predictions
    # In a real implementation, we'd map template positions to text positions
    tokens = text.split()
    
    # Simple rule: First 2 tokens are PII (obviously wrong, just for demo)
    predictions = ['O'] * len(tokens)
    if len(tokens) >= 2:
        predictions[0] = 'B-PERSON'  # Assume first token is beginning of PII
        predictions[1] = 'I-PERSON'  # Assume second token is inside PII
    
    return predictions

def evaluate_baseline(df, split_name="test"):
    """Evaluate baseline predictions"""
    print(f"\n📊 Evaluating baseline on {split_name} set...")
    
    # For simplicity, we'll use a heuristic since we don't have templates
    # In reality, you'd need the original templates with markers
    
    results = []
    all_true = []
    all_pred = []
    
    # Simple heuristic baseline:
    # 1. Any capitalized word might be a name
    # 2. Words with @ are emails
    # 3. Number patterns are phones/SSNs
    
    for idx, row in df.iterrows():
        text = row['text']
        true_labels = row['labels']
        
        # Tokenize text (simple split)
        tokens = text.split()
        
        # Create predictions with simple rules
        pred_labels = []
        for token in tokens:
            # Rule 1: Capitalized words might be names
            if token[0].isupper() and len(token) > 1:
                if token.isupper():  # All caps might be acronym
                    pred_labels.append('O')
                else:
                    pred_labels.append('B-PERSON')  # Start of name
            # Rule 2: Contains @ → email
            elif '@' in token and '.' in token:
                pred_labels.append('B-EMAIL')
            # Rule 3: Looks like a number sequence
            elif re.match(r'\d{3}[-.]?\d{3}[-.]?\d{4}', token):  # Phone
                pred_labels.append('B-PHONE')
            elif re.match(r'\d{3}-\d{2}-\d{4}', token):  # SSN
                pred_labels.append('B-SSN')
            else:
                pred_labels.append('O')
        
        # Ensure same length as true labels
        if len(pred_labels) != len(true_labels):
            # Pad or truncate
            if len(pred_labels) < len(true_labels):
                pred_labels.extend(['O'] * (len(true_labels) - len(pred_labels)))
            else:
                pred_labels = pred_labels[:len(true_labels)]
        
        # For evaluation, flatten BIO to binary (PII vs not-PII)
        true_binary = [1 if label != 'O' else 0 for label in true_labels]
        pred_binary = [1 if label != 'O' else 0 for label in pred_labels]
        
        all_true.extend(true_binary)
        all_pred.extend(pred_binary)
        
        # Calculate sample-level metrics
        if len(true_binary) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_binary, pred_binary, average='binary', zero_division=0
            )
            results.append({
                'text': text[:50] + "...",  # First 50 chars
                'true_pii': sum(true_binary),
                'pred_pii': sum(pred_binary),
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
    
    # Overall metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_pred, average='binary', zero_division=0
    )
    
    overall = {
        'split': split_name,
        'samples': len(df),
        'total_tokens': len(all_true),
        'true_pii_tokens': sum(all_true),
        'pred_pii_tokens': sum(all_pred),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }
    
    print(f"\n📈 Baseline Results ({split_name}):")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1:.3f}")
    print(f"   PII tokens: {sum(all_true)} true, {sum(all_pred)} predicted")
    
    return overall, results[:5]  # Return first 5 examples

def save_results(results, split_name):
    """Save evaluation results"""
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = output_dir / f"baseline_{split_name}_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filename}")
    return filename

def analyze_errors(df, predictions, true_labels):
    """Analyze common error patterns"""
    print("\n🔍 Error Analysis:")
    
    # Count error types
    error_counts = {
        'false_positives': 0,  # Predicted PII but not actually PII
        'false_negatives': 0,  # Missed real PII
        'correct': 0
    }
    
    for true, pred in zip(true_labels, predictions):
        if pred != 'O' and true == 'O':
            error_counts['false_positives'] += 1
        elif pred == 'O' and true != 'O':
            error_counts['false_negatives'] += 1
        else:
            error_counts['correct'] += 1
    
    total = sum(error_counts.values())
    print(f"   Correct predictions: {error_counts['correct']} ({error_counts['correct']/total*100:.1f}%)")
    print(f"   False positives: {error_counts['false_positives']} ({error_counts['false_positives']/total*100:.1f}%)")
    print(f"   False negatives: {error_counts['false_negatives']} ({error_counts['false_negatives']/total*100:.1f}%)")
    
    return error_counts

def create_baseline_summary(train_results, val_results, test_results):
    """Create summary report of baseline performance"""
    summary = {
        'baseline_type': 'simple_heuristic',
        'heuristics_used': [
            'Capitalized words → PERSON',
            'Contains @ and . → EMAIL',
            'Phone pattern → PHONE',
            'SSN pattern → SSN'
        ],
        'performance': {
            'train': train_results[0],
            'val': val_results[0],
            'test': test_results[0]
        },
        'notes': 'Simple rule-based baseline for comparison with ML model'
    }
    
    output_dir = Path("outputs/results")
    with open(output_dir / "baseline_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Summary saved to: {output_dir}/baseline_summary.json")
    
    # Print comparison table
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE SUMMARY")
    print("="*60)
    for split in ['train', 'val', 'test']:
        perf = summary['performance'][split]
        print(f"{split.upper():6} | Precision: {perf['precision']:.3f} | "
              f"Recall: {perf['recall']:.3f} | F1: {perf['f1_score']:.3f}")
    print("="*60)
    
    return summary

def main():
    """Main baseline evaluation pipeline"""
    print("="*60)
    print("🎯 PHASE 3: BASELINE MODEL EVALUATION")
    print("="*60)
    
    # Load data
    train_df, val_df, test_df = load_data()
    
    # Evaluate on all splits
    print("\n" + "="*40)
    train_results = evaluate_baseline(train_df, "train")
    save_results(train_results, "train")
    
    print("\n" + "="*40)
    val_results = evaluate_baseline(val_df, "val")
    save_results(val_results, "val")
    
    print("\n" + "="*40)
    test_results = evaluate_baseline(test_df, "test")
    save_results(test_results, "test")
    
    # Create summary
    summary = create_baseline_summary(train_results, val_results, test_results)
    
    print("\n" + "="*60)
    print("✅ PHASE 3 COMPLETE")
    print("="*60)
    print("\n📊 Baseline established for ML model comparison.")
    print("📁 Results saved in outputs/results/")
    print("\nNext: Phase 4 - Train DP models")
    
    return summary

if __name__ == "__main__":
    main()
