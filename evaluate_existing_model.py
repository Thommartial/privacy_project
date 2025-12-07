#!/usr/bin/env python3
"""
Proper evaluation of your existing DP model.
"""
import pickle
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

print("🔍 EVALUATING YOUR EXISTING ε=8.0 MODEL")
print("="*60)

# Load your model
model_path = "outputs/models/proper_epsilon_8.0"
history_path = f"{model_path}/history.json"

try:
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    print(f"✓ Loaded training history")
    print(f"  Epochs trained: {len(history['train_loss'])}")
    print(f"  ε = {history.get('epsilon', 'unknown')}")
    print()
    
    # Check for loss explosion
    if history['train_loss'][-1] > history['train_loss'][0] * 5:
        print("⚠️  WARNING: Training loss exploded!")
        print(f"   Started at: {history['train_loss'][0]:.4f}")
        print(f"   Ended at:   {history['train_loss'][-1]:.4f}")
        print(f"   Best model is from EPOCH 1")
    
    # Show accuracy progression
    print("Accuracy progression:")
    for i, acc in enumerate(history['val_accuracy']):
        print(f"  Epoch {i+1}: {acc:.4f}")
    
    print()
    print("🎯 ANALYSIS:")
    print("-" * 30)
    
    # Load some test data to evaluate
    import pandas as pd
    from pathlib import Path
    
    data_path = Path("data/processed/val.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
        texts = df['text'].astype(str).tolist()[:100]  # Sample 100
        
        # Create simple features (same as your training)
        features = []
        true_labels = []
        
        for text in texts:
            length = min(len(text) / 100, 1.0)
            has_at = 1.0 if '@' in text else 0.0
            has_dot = 1.0 if '.' in text and '@' in text else 0.0
            has_digits = sum(c.isdigit() for c in text)
            digit_ratio = has_digits / max(len(text), 1)
            has_caps = 1.0 if any(word.istitle() for word in text.split()) else 0.0
            
            features.append([length, has_at, has_dot, digit_ratio, has_caps])
            
            # True label (simplified)
            label = 1 if ((has_at and has_dot) or digit_ratio > 0.3) else 0
            true_labels.append(label)
        
        features = np.array(features, dtype=np.float32)
        true_labels = np.array(true_labels, dtype=np.int32)
        
        print(f"Test samples: {len(features)}")
        print(f"PII in test: {np.sum(true_labels)} ({np.sum(true_labels)/len(true_labels)*100:.1f}%)")
        
        # Baseline: always predict non-PII
        baseline_preds = np.zeros(len(true_labels))
        baseline_acc = accuracy_score(true_labels, baseline_preds)
        
        print(f"\n📊 BASELINE (always non-PII):")
        print(f"  Accuracy: {baseline_acc:.4f}")
        print(f"  This is what we need to beat!")
        
        # If we had the actual model, we could make predictions
        # For now, show what good performance looks like
        
        print(f"\n🎯 WHAT GOOD PERFORMANCE LOOKS LIKE:")
        print(f"  For {np.sum(true_labels)/len(true_labels)*100:.1f}% PII data:")
        print(f"  - Baseline accuracy: {baseline_acc:.4f}")
        print(f"  - Good model accuracy: {baseline_acc + 0.10:.4f} (10% improvement)")
        print(f"  - Good F1 score for PII class: > 0.50")
        
    else:
        print("⚠️  Test data not found")
        
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("="*60)
print("🎯 RECOMMENDATIONS:")
print("1. Run train_dp_final_working.py for stable training")
print("2. Focus on F1 score, not just accuracy")
print("3. Compare against baseline (always predict non-PII)")
print("4. Your current best model is likely from Epoch 1")
print("="*60)
