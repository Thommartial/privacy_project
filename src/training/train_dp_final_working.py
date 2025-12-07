#!/usr/bin/env python3
"""
FINAL WORKING VERSION - Simple, stable DP training.
No dropout, no batch norm, just clean training.
"""
import os
import sys
import argparse
import time
from pathlib import Path
from collections import Counter

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from jax import tree_util

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

np.random.seed(42)

# ========== SIMPLEST POSSIBLE MODEL ==========
class FinalDPModel(nn.Module):
    """Simplest possible model - no dropout, no normalization."""
    hidden_size: int
    num_classes: int = 2
    
    @nn.compact
    def __call__(self, x):
        # Simple feedforward
        x = nn.Dense(self.hidden_size)(x)
        x = jnp.tanh(x)  # tanh is more stable than relu
        
        x = nn.Dense(self.hidden_size // 2)(x)
        x = jnp.tanh(x)
        
        x = nn.Dense(self.num_classes)(x)
        return x

# ========== SIMPLE DP NOISE ==========
def add_dp_noise_final(grads, epsilon, delta, clip_norm=0.3):
    """Simple DP noise."""
    def clip(g):
        norm = jnp.linalg.norm(g) + 1e-10
        scale = jnp.minimum(1.0, clip_norm / norm)
        return g * scale
    
    grads = tree_util.tree_map(clip, grads)
    
    # Noise calculation
    safe_epsilon = jnp.maximum(epsilon, 1.0)
    noise_scale = clip_norm * jnp.sqrt(2 * jnp.log(1.25 / delta)) / safe_epsilon
    
    key = jax.random.PRNGKey(42)
    
    def add_noise(g):
        nonlocal key
        key, subkey = jax.random.split(key)
        return g + noise_scale * jax.random.normal(subkey, g.shape)
    
    return tree_util.tree_map(add_noise, grads)

# ========== LOSS FUNCTION ==========
def compute_loss(logits, labels):
    """Simple cross-entropy loss."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

# ========== TRAINING ==========
@jax.jit
def train_step(state, batch, labels, epsilon, delta):
    """Training step."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch)
        return compute_loss(logits, labels)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    grads = add_dp_noise_final(grads, epsilon, delta)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@jax.jit
def eval_step(state, batch, labels):
    """Evaluation step."""
    logits = state.apply_fn({'params': state.params}, batch)
    loss = compute_loss(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    return loss, preds

# ========== DATA ==========
def load_final_features(max_samples=5000):
    """Load and create features."""
    data_path = project_root / "data/processed"
    
    if (data_path / "train.csv").exists():
        import pandas as pd
        import re
        
        df = pd.read_csv(data_path / "train.csv")
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)
        
        texts = df['text'].astype(str).tolist()
        features = []
        labels = []
        
        for text in texts[:max_samples] if max_samples else texts:
            # Simple features only
            length = min(len(text) / 50, 2.0)
            
            # Email
            has_at = 1.0 if '@' in text else 0.0
            has_dot_after_at = 1.0 if '@' in text and '.' in text.split('@')[-1] else 0.0
            
            # Numbers
            digits = sum(c.isdigit() for c in text)
            digit_ratio = digits / max(len(text), 1)
            
            # Capitals
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
            
            # Simple labeling
            label = 1 if ((has_at and has_dot_after_at) or digit_ratio > 0.3) else 0
            labels.append(label)
        
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        print(f"📊 Loaded {len(features)} samples")
        print(f"   PII: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.1f}%)")
        print(f"   Features: {features.shape[1]} dimensions")
        
        return features, labels, features.shape[1]
    else:
        # Synthetic data
        n = max_samples or 1000
        X = np.random.randn(n, 6).astype(np.float32)
        y = (np.random.rand(n) > 0.82).astype(np.int32)
        return X, y, 6

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=8.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=32)  # Very small
    parser.add_argument('--learning_rate', type=float, default=0.0005)  # Small
    parser.add_argument('--max_samples', type=int, default=5000)
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"🚀 FINAL DP MODEL TRAINING (ε={args.epsilon})")
    print("="*70)
    
    # Setup
    model_dir = Path("outputs/models") / f"final_epsilon_{args.epsilon}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load
    X, y, input_dim = load_final_features(args.max_samples)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"\n📈 Dataset:")
    print(f"   Total: {len(X)} samples")
    print(f"   Train: {len(X_train)} ({np.sum(y_train)} PII, {np.sum(y_train)/len(y_train)*100:.1f}%)")
    print(f"   Val:   {len(X_val)} ({np.sum(y_val)} PII, {np.sum(y_val)/len(y_val)*100:.1f}%)")
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    model = FinalDPModel(hidden_size=args.hidden_size)
    dummy = jnp.ones((1, input_dim))
    params = model.init(rng, dummy)['params']
    
    optimizer = optax.adam(args.learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # Convert
    epsilon_jax = jnp.array(args.epsilon, dtype=jnp.float32)
    delta_jax = jnp.array(args.delta, dtype=jnp.float32)
    
    # Train
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    print("\n🏋️‍♂️ Starting training...")
    print(f"   Model: {args.hidden_size} → {args.hidden_size//2} → 2")
    print(f"   LR: {args.learning_rate}, Batch: {args.batch_size}")
    print("="*50)
    
    start = time.time()
    best_epoch = 0
    best_f1 = 0
    patience = 0
    max_patience = 5
    
    for epoch in range(args.epochs):
        # Shuffle
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        
        epoch_loss = 0
        batches = 0
        
        # Train
        for i in range(0, len(X_shuffled), args.batch_size):
            batch = jnp.array(X_shuffled[i:i+args.batch_size])
            labels = jnp.array(y_shuffled[i:i+args.batch_size])
            state, loss = train_step(state, batch, labels, epsilon_jax, delta_jax)
            epoch_loss += float(loss)
            batches += 1
        
        avg_train_loss = epoch_loss / max(batches, 1)
        
        # Validate
        val_batch = jnp.array(X_val)
        val_labels = jnp.array(y_val)
        val_loss, val_preds = eval_step(state, val_batch, val_labels)
        val_preds_np = np.array(val_preds)
        
        # Metrics
        val_acc = accuracy_score(y_val, val_preds_np)
        val_precision = precision_score(y_val, val_preds_np, zero_division=0)
        val_recall = recall_score(y_val, val_preds_np, zero_division=0)
        val_f1 = f1_score(y_val, val_preds_np, zero_division=0)
        
        # Store
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Early stopping on F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            patience = 0
            
            # Save best model
            best_params = tree_util.tree_map(lambda x: np.array(x), state.params)
            import pickle
            with open(model_dir / "best_params.pkl", 'wb') as f:
                pickle.dump(best_params, f)
        else:
            patience += 1
        
        # Print
        print(f"Epoch {epoch+1:2d}/{args.epochs}: "
              f"Loss={avg_train_loss:.4f}→{float(val_loss):.4f} | "
              f"Acc={val_acc:.3f} | "
              f"P={val_precision:.3f} R={val_recall:.3f} F1={val_f1:.3f} | "
              f"Best@{best_epoch+1} (F1={best_f1:.3f})")
        
        if patience >= max_patience:
            print(f"\n⚠️  Early stopping - no improvement for {max_patience} epochs")
            break
    
    print(f"\n✅ Training completed in {time.time()-start:.1f}s")
    print(f"🎯 Best F1: {best_f1:.4f} at epoch {best_epoch+1}")
    
    # Load best model for final evaluation
    with open(model_dir / "best_params.pkl", 'rb') as f:
        best_params = pickle.load(f)
    
    # Create state with best params
    best_state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=jax.tree_util.tree_map(jnp.array, best_params),
        tx=optimizer
    )
    
    # Final evaluation
    val_batch = jnp.array(X_val)
    val_labels_jax = jnp.array(y_val)
    final_loss, final_preds = eval_step(best_state, val_batch, val_labels_jax)
    final_preds_np = np.array(final_preds)
    
    print("\n" + "="*70)
    print("📊 FINAL EVALUATION (Best Model):")
    print("="*70)
    
    print(f"Loss: {float(final_loss):.4f}")
    print(f"Accuracy: {accuracy_score(y_val, final_preds_np):.4f}")
    print(f"F1 Score: {f1_score(y_val, final_preds_np, zero_division=0):.4f}")
    print(f"Precision: {precision_score(y_val, final_preds_np, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val, final_preds_np, zero_division=0):.4f}")
    print()
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_val, final_preds_np)
    print(f"            Predicted")
    print(f"            Non-PII  PII")
    print(f"Actual Non-PII  {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"Actual PII      {cm[1,0]:4d}    {cm[1,1]:4d}")
    print()
    
    print("Classification Report:")
    print(classification_report(y_val, final_preds_np, target_names=['Non-PII', 'PII']))
    
    # Baseline comparison
    baseline_acc = np.mean(y_val == 0)  # Always predict non-PII
    print(f"\n📊 BASELINE (Always predict Non-PII):")
    print(f"   Accuracy: {baseline_acc:.4f}")
    print(f"   Your model improvement: {accuracy_score(y_val, final_preds_np) - baseline_acc:.4f}")
    
    # Save everything
    import json
    
    with open(model_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    with open(model_dir / "config.json", 'w') as f:
        config = vars(args)
        config.update({
            'input_dim': input_dim,
            'best_epoch': best_epoch,
            'best_f1': float(best_f1),
            'final_accuracy': float(accuracy_score(y_val, final_preds_np)),
            'baseline_accuracy': float(baseline_acc)
        })
        json.dump(config, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    epochs_range = list(range(1, len(history['train_loss']) + 1))
    
    # Loss
    axes[0, 0].plot(epochs_range, history['train_loss'], 'b-o', label='Train', alpha=0.7)
    axes[0, 0].plot(epochs_range, history['val_loss'], 'r-s', label='Val', alpha=0.7)
    axes[0, 0].axvline(x=best_epoch+1, color='g', linestyle='--', label=f'Best (epoch {best_epoch+1})')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs_range, history['val_acc'], 'g-o')
    axes[0, 1].axvline(x=best_epoch+1, color='g', linestyle='--')
    axes[0, 1].axhline(y=baseline_acc, color='r', linestyle=':', label=f'Baseline ({baseline_acc:.3f})')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 2].plot(epochs_range, history['val_f1'], 'purple')
    axes[0, 2].axvline(x=best_epoch+1, color='g', linestyle='--')
    axes[0, 2].set_title('F1 Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 0].plot(epochs_range, history['val_precision'], 'orange', label='Precision')
    axes[1, 0].plot(epochs_range, history['val_recall'], 'brown', label='Recall')
    axes[1, 0].axvline(x=best_epoch+1, color='g', linestyle='--')
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Confusion matrix visualization
    axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_xticklabels(['Non-PII', 'PII'])
    axes[1, 1].set_yticklabels(['Non-PII', 'PII'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black")
    
    # Summary text
    summary_text = f"""
    Best Epoch: {best_epoch+1}
    F1 Score: {best_f1:.4f}
    Accuracy: {history['val_acc'][best_epoch]:.4f}
    Baseline: {baseline_acc:.4f}
    Improvement: {history['val_acc'][best_epoch] - baseline_acc:.4f}
    
    Final Results:
    - Precision: {history['val_precision'][best_epoch]:.4f}
    - Recall: {history['val_recall'][best_epoch]:.4f}
    - F1: {history['val_f1'][best_epoch]:.4f}
    """
    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'DP Model Training Results (ε={args.epsilon})', fontsize=16)
    plt.tight_layout()
    plt.savefig(model_dir / "training_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create final report
    report = f"""
    ======================================================================
    📊 FINAL DP MODEL TRAINING REPORT (ε={args.epsilon})
    ======================================================================
    
    DATASET:
    - Total samples: {len(X)}
    - PII rate: {np.sum(y)/len(y)*100:.1f}%
    - Train/Validation: {len(X_train)} / {len(X_val)}
    
    MODEL:
    - Architecture: {input_dim} → {args.hidden_size} → {args.hidden_size//2} → 2
    - Learning rate: {args.learning_rate}
    - Batch size: {args.batch_size}
    
    RESULTS:
    - Best epoch: {best_epoch + 1}
    - Best F1 score: {best_f1:.4f}
    - Validation accuracy: {history['val_acc'][best_epoch]:.4f}
    - Baseline accuracy (always predict non-PII): {baseline_acc:.4f}
    - Improvement over baseline: {history['val_acc'][best_epoch] - baseline_acc:.4f}
    
    PRIVACY:
    - ε (privacy budget): {args.epsilon}
    - δ: {args.delta}
    - Training samples: {len(X_train)}
    - Epochs trained: {len(history['train_loss'])}
    
    FILES SAVED:
    - Model parameters: {model_dir}/best_params.pkl
    - Training history: {model_dir}/history.json
    - Configuration: {model_dir}/config.json
    - Results plot: {model_dir}/training_results.png
    
    ======================================================================
    """
    
    print(report)
    
    with open(model_dir / "final_report.txt", 'w') as f:
        f.write(report)
    
    print(f"📁 All results saved to: {model_dir}")
    print("="*70)
    print("🎉 TRAINING COMPLETE! Your model is ready for evaluation.")
    
    return model_dir

if __name__ == "__main__":
    main()
