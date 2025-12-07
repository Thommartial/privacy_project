#!/usr/bin/env python3
"""
Generic evaluation script for DP models.
Takes epsilon as parameter and evaluates the corresponding model.
"""
import os
import sys
import argparse
import pickle
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
from jax import tree_util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Define the model architecture (must match training)
class EvaluationDPModel(nn.Module):
    """Model architecture for evaluation (must match training)."""
    hidden_size: int
    num_classes: int = 2
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = jnp.tanh(x)
        x = nn.Dense(self.hidden_size // 2)(x)
        x = jnp.tanh(x)
        x = nn.Dense(self.num_classes)(x)
        return x

def load_test_data(max_samples=1000):
    """Load test data for evaluation."""
    data_path = project_root / "data/processed"
    
    if (data_path / "test.csv").exists():
        import pandas as pd
        import re
        
        df = pd.read_csv(data_path / "test.csv")
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)
        
        texts = df['text'].astype(str).tolist()
        features = []
        true_labels = []
        
        for text in texts:
            # Same feature extraction as training
            length = min(len(text) / 50, 2.0)
            
            has_at = 1.0 if '@' in text else 0.0
            has_dot_after_at = 1.0 if '@' in text and '.' in text.split('@')[-1] else 0.0
            
            digits = sum(c.isdigit() for c in text)
            digit_ratio = digits / max(len(text), 1)
            
            words = text.split()
            has_caps = 1.0 if any(w.istitle() for w in words) else 0.0
            
            features.append([
                length,
                has_at,
                has_dot_after_at,
                digit_ratio,
                has_caps,
                len(words) / 20.0
            ])
            
            # True label
            label = 1 if ((has_at and has_dot_after_at) or digit_ratio > 0.3) else 0
            true_labels.append(label)
        
        features = np.array(features, dtype=np.float32)
        true_labels = np.array(true_labels, dtype=np.int32)
        
        # Scale features (use saved scaler if available, otherwise fit new)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        print(f"📊 Loaded {len(features)} test samples")
        print(f"   PII: {np.sum(true_labels)} ({np.sum(true_labels)/len(true_labels)*100:.1f}%)")
        
        return features, true_labels, features.shape[1]
    else:
        # Fallback: use validation split
        print("⚠️  Test data not found, using validation split")
        return load_validation_data(max_samples)

def load_validation_data(max_samples=1000):
    """Load validation data."""
    data_path = project_root / "data/processed"
    
    if (data_path / "val.csv").exists():
        import pandas as pd
        df = pd.read_csv(data_path / "val.csv")
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)
        
        texts = df['text'].astype(str).tolist()
        features = []
        true_labels = []
        
        for text in texts:
            length = min(len(text) / 50, 2.0)
            has_at = 1.0 if '@' in text else 0.0
            has_dot_after_at = 1.0 if '@' in text and '.' in text.split('@')[-1] else 0.0
            digits = sum(c.isdigit() for c in text)
            digit_ratio = digits / max(len(text), 1)
            words = text.split()
            has_caps = 1.0 if any(w.istitle() for w in words) else 0.0
            
            features.append([
                length,
                has_at,
                has_dot_after_at,
                digit_ratio,
                has_caps,
                len(words) / 20.0
            ])
            
            label = 1 if ((has_at and has_dot_after_at) or digit_ratio > 0.3) else 0
            true_labels.append(label)
        
        features = np.array(features, dtype=np.float32)
        true_labels = np.array(true_labels, dtype=np.int32)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features, true_labels, features.shape[1]
    else:
        # Generate synthetic data
        np.random.seed(42)
        n = max_samples or 500
        X = np.random.randn(n, 6).astype(np.float32)
        y = (np.random.rand(n) > 0.82).astype(np.int32)
        return X, y, 6

def evaluate_model(model_dir, output_dir=None):
    """Evaluate a trained DP model."""
    print("="*70)
    print(f"🔍 EVALUATING DP MODEL from: {model_dir}")
    print("="*70)
    
    # Load model configuration
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        # Try to infer epsilon from directory name
        eps = float(Path(model_dir).name.split('_')[-1])
        config = {'epsilon': eps, 'hidden_size': 32, 'input_dim': 6}
        print(f"   Inferred ε={eps} from directory name")
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    epsilon = config.get('epsilon', 'unknown')
    hidden_size = config.get('hidden_size', 32)
    input_dim = config.get('input_dim', 6)
    
    # Load model parameters
    params_path = Path(model_dir) / "best_params.pkl"
    if not params_path.exists():
        params_path = Path(model_dir) / "params.pkl"
    
    if not params_path.exists():
        print(f"❌ Model parameters not found in {model_dir}")
        return None
    
    with open(params_path, 'rb') as f:
        params = pickle.load(f)
    
    # Load training history if available
    history_path = Path(model_dir) / "history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        best_epoch = history.get('best_epoch', 0)
        best_f1 = max(history.get('val_f1', [0]))
    else:
        history = None
        best_epoch = 0
        best_f1 = 0
    
    # Create model
    model = EvaluationDPModel(hidden_size=hidden_size)
    
    # Load test data
    X_test, y_test, actual_input_dim = load_test_data(max_samples=1000)
    
    if actual_input_dim != input_dim:
        print(f"⚠️  Input dimension mismatch: config={input_dim}, data={actual_input_dim}")
        print("   Using data dimension...")
        input_dim = actual_input_dim
    
    # Make predictions
    def predict(params, x):
        logits = model.apply({'params': params}, x)
        return jnp.argmax(logits, axis=-1), logits
    
    y_pred, logits = predict(params, X_test)
    y_pred_np = np.array(y_pred)
    logits_np = np.array(logits)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred_np)
    precision = precision_score(y_test, y_pred_np, zero_division=0)
    recall = recall_score(y_test, y_pred_np, zero_division=0)
    f1 = f1_score(y_test, y_pred_np, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_np)
    
    # Baseline (always predict non-PII)
    baseline_acc = 1.0 - np.mean(y_test)
    
    # Print results
    print("\n" + "="*70)
    print(f"📊 EVALUATION RESULTS (ε={epsilon})")
    print("="*70)
    
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"  • Accuracy:      {accuracy:.4f}")
    print(f"  • Precision:     {precision:.4f}")
    print(f"  • Recall:        {recall:.4f}")
    print(f"  • F1 Score:      {f1:.4f}")
    print(f"  • Baseline Acc:  {baseline_acc:.4f} (always predict non-PII)")
    print(f"  • Improvement:   {accuracy - baseline_acc:.4f}")
    
    if history:
        print(f"  • Best Train F1: {best_f1:.4f} (epoch {best_epoch+1})")
    
    print(f"\n🔢 CONFUSION MATRIX:")
    print(f"            Predicted")
    print(f"            Non-PII  PII")
    print(f"Actual Non-PII  {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"Actual PII      {cm[1,0]:6d}    {cm[1,1]:6d}")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred_np, target_names=['Non-PII', 'PII']))
    
    # Create output directory
    if output_dir:
        eval_dir = Path(output_dir) / f"eval_epsilon_{epsilon}"
    else:
        eval_dir = Path(model_dir) / "evaluation"
    
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Save evaluation results
    results = {
        'epsilon': epsilon,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'baseline_accuracy': float(baseline_acc),
            'improvement': float(accuracy - baseline_acc)
        },
        'confusion_matrix': cm.tolist(),
        'dataset_info': {
            'test_samples': len(X_test),
            'pii_count': int(np.sum(y_test)),
            'pii_percentage': float(np.mean(y_test))
        }
    }
    
    results_path = eval_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Create evaluation plots
    create_evaluation_plots(X_test, y_test, y_pred_np, logits_np, eval_dir, epsilon)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    
    return results

def create_evaluation_plots(X_test, y_test, y_pred, logits, output_dir, epsilon):
    """Create evaluation plots."""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title(f'Confusion Matrix (ε={epsilon})', fontsize=12)
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Non-PII', 'PII'])
    ax1.set_yticklabels(['Non-PII', 'PII'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # 2. ROC Curve
    ax2 = plt.subplot(2, 3, 2)
    if logits.shape[1] == 2:
        # Get probabilities for positive class
        probs = np.exp(logits[:, 1]) / np.sum(np.exp(logits), axis=1)
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    ax3 = plt.subplot(2, 3, 3)
    if logits.shape[1] == 2:
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
        ax3.plot(recall_vals, precision_vals, color='green', lw=2)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision-Recall Curve')
        ax3.grid(True, alpha=0.3)
    
    # 4. Metrics Bar Chart
    ax4 = plt.subplot(2, 3, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0)
    ]
    
    colors = ['blue', 'green', 'orange', 'red']
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.set_ylim([0, 1.1])
    ax4.set_title('Performance Metrics')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom')
    
    # 5. Class Distribution
    ax5 = plt.subplot(2, 3, 5)
    unique, counts = np.unique(y_test, return_counts=True)
    ax5.bar(['Non-PII', 'PII'], counts, color=['blue', 'orange'], alpha=0.7)
    ax5.set_title('Test Set Class Distribution')
    ax5.set_ylabel('Count')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for i, count in enumerate(counts):
        ax5.text(i, count + max(counts)*0.02, str(count), ha='center')
    
    # 6. Summary Text
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    baseline = 1.0 - np.mean(y_test)
    
    summary_text = f"""
    ε (Privacy Budget): {epsilon}
    
    Test Set:
    - Samples: {len(y_test)}
    - PII: {np.sum(y_test)} ({np.mean(y_test)*100:.1f}%)
    
    Results:
    - Accuracy: {accuracy:.4f}
    - Precision: {precision:.4f}
    - Recall: {recall:.4f}
    - F1 Score: {f1:.4f}
    
    Baseline (always non-PII):
    - Accuracy: {baseline:.4f}
    - Improvement: {accuracy - baseline:.4f}
    
    Confusion Matrix:
    TN={cm[0,0]}, FP={cm[0,1]}
    FN={cm[1,0]}, TP={cm[1,1]}
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, 
            verticalalignment='center', family='monospace')
    
    plt.suptitle(f'DP Model Evaluation (ε={epsilon})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_dir / "evaluation_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Evaluation plots saved to: {plot_path}")

def compare_multiple_models(epsilons=[8.0, 5.0, 3.0, 2.0, 1.0, 0.5], output_dir=None, with_baseline=False):
    """Compare multiple DP models with different epsilon values."""
    print("="*70)
    print("📊 COMPARING MULTIPLE DP MODELS")
    print("="*70)
    
    results = {}
    
    for eps in epsilons:
        model_dir = project_root / f"outputs/models/final_epsilon_{eps}"
        if model_dir.exists():
            print(f"\n🔍 Evaluating ε={eps}...")
            try:
                result = evaluate_model(model_dir, output_dir)
                if result:
                    results[eps] = result
            except Exception as e:
                print(f"   ❌ Error evaluating ε={eps}: {e}")
        else:
            print(f"   ⚠️  Model for ε={eps} not found")
    
    if not results:
        print("\n❌ No models found for comparison")
        return
    
    # Create comparison plot
    create_comparison_plot(results, output_dir)
    
    return results

def create_comparison_plot(results, output_dir=None):
    """Create comparison plot for multiple models."""
    epsilons = sorted(results.keys())
    
    # Extract metrics
    accuracies = [results[eps]['metrics']['accuracy'] for eps in epsilons]
    f1_scores = [results[eps]['metrics']['f1_score'] for eps in epsilons]
    precisions = [results[eps]['metrics']['precision'] for eps in epsilons]
    recalls = [results[eps]['metrics']['recall'] for eps in epsilons]
    baselines = [results[eps]['metrics']['baseline_accuracy'] for eps in epsilons]
    
    # Create plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Accuracy vs Epsilon
    ax1.semilogx(epsilons, accuracies, 'bo-', linewidth=2, markersize=8, label='DP Model')
    if baselines:
        ax1.axhline(y=baselines[0], color='r', linestyle='--', linewidth=2, label='Baseline')
    ax1.set_xlabel('ε (Privacy Budget) → More Private', fontsize=11)
    ax1.set_ylabel('Accuracy', fontsize=11)
    ax1.set_title('Privacy vs Accuracy Tradeoff', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0.7, 1.05)
    
    # Add annotations
    for i, eps in enumerate(epsilons):
        ax1.annotate(f'ε={eps}', (eps, accuracies[i]), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # 2. F1 Score vs Epsilon
    ax2.semilogx(epsilons, f1_scores, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('ε (Privacy Budget) → More Private', fontsize=11)
    ax2.set_ylabel('F1 Score', fontsize=11)
    ax2.set_title('Privacy vs F1 Score Tradeoff', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.7, 1.05)
    
    # 3. Precision & Recall vs Epsilon
    ax3.semilogx(epsilons, precisions, 'orange', marker='o', linewidth=2, markersize=8, label='Precision')
    ax3.semilogx(epsilons, recalls, 'purple', marker='s', linewidth=2, markersize=8, label='Recall')
    ax3.set_xlabel('ε (Privacy Budget) → More Private', fontsize=11)
    ax3.set_ylabel('Score', fontsize=11)
    ax3.set_title('Precision & Recall vs Privacy', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0.7, 1.05)
    
    # 4. Improvement over Baseline
    improvements = [acc - base for acc, base in zip(accuracies, baselines)]
    ax4.bar(range(len(epsilons)), improvements, color=['blue', 'green', 'orange', 'red', 'purple', 'brown'])
    ax4.set_xlabel('Privacy Level', fontsize=11)
    ax4.set_ylabel('Improvement over Baseline', fontsize=11)
    ax4.set_title('Performance Improvement vs Baseline', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(epsilons)))
    ax4.set_xticklabels([f'ε={eps}' for eps in epsilons], rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, imp in enumerate(improvements):
        ax4.text(i, imp + 0.01, f'{imp:.3f}', ha='center', va='bottom')
    
    plt.suptitle('Differential Privacy: Privacy-Performance Tradeoff Analysis', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_dir:
        comp_dir = Path(output_dir) / "comparison"
    else:
        comp_dir = project_root / "outputs" / "comparison"
    
    comp_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = comp_dir / "privacy_tradeoff_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 Comparison plot saved to: {plot_path}")
    
    # Create summary table
    print("\n" + "="*70)
    print("📊 SUMMARY: PRIVACY-PERFORMANCE TRADEOFF")
    print("="*70)
    print("ε (Privacy) | Accuracy | F1 Score | Precision | Recall | Improvement")
    print("-" * 70)
    
    for eps in epsilons:
        res = results[eps]['metrics']
        print(f"{eps:10.1f} | {res['accuracy']:8.4f} | {res['f1_score']:9.4f} | "
              f"{res['precision']:9.4f} | {res['recall']:6.4f} | {res['improvement']:11.4f}")
    
    print("\n✅ Comparison complete!")

def main():
    parser = argparse.ArgumentParser(description='Evaluate DP model')
    parser.add_argument('--epsilon', type=float, help='Epsilon value of model to evaluate')
    parser.add_argument('--model_dir', type=str, help='Direct path to model directory')
    parser.add_argument('--compare_all', action='store_true', help='Compare all trained models')
    parser.add_argument('--with_baseline', action='store_true', help='Include baseline in comparison')

    parser.add_argument('--epsilons', type=float, nargs='+', default=[8.0, 5.0, 3.0, 2.0, 1.0, 0.5],
                       help='List of epsilons to compare')
    parser.add_argument('--output_dir', type=str, default='outputs/evaluation',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare_all:
        # Compare multiple models
        compare_multiple_models(args.epsilons, output_dir, args.with_baseline)
    elif args.model_dir:
        # Evaluate specific model directory
        evaluate_model(args.model_dir, output_dir)
    elif args.epsilon:
        # Evaluate model by epsilon value
        model_dir = project_root / f"outputs/models/final_epsilon_{args.epsilon}"
        if not model_dir.exists():
            # Try alternative naming
            model_dir = project_root / f"outputs/models/proper_epsilon_{args.epsilon}"
        
        if model_dir.exists():
            evaluate_model(model_dir, output_dir)
        else:
            print(f"❌ Model directory not found for ε={args.epsilon}")
            print(f"   Checked: {model_dir}")
            print("   Available models:")
            models_dir = project_root / "outputs/models"
            if models_dir.exists():
                for item in models_dir.iterdir():
                    if item.is_dir():
                        print(f"   - {item.name}")
    else:
        print("❌ Please specify either --epsilon, --model_dir, or --compare_all")
        print("\nUsage examples:")
        print("  Evaluate ε=8.0 model: python evaluate_dp_model.py --epsilon 8.0")
        print("  Evaluate specific model: python evaluate_dp_model.py --model_dir outputs/models/final_epsilon_8.0")
        print("  Compare all models: python evaluate_dp_model.py --compare_all")
        print("  Compare specific epsilons: python evaluate_dp_model.py --compare_all --epsilons 8.0 2.0 0.5")

if __name__ == "__main__":
    main()
