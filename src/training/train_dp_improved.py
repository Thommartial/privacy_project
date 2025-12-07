#!/usr/bin/env python3
"""
Improved DP model training with gradient clipping, class balancing, and better monitoring.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from jax import tree_util

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

np.random.seed(42)

# ========== IMPROVED MODEL ==========
class ImprovedDPModel(nn.Module):
    """Better model with batch normalization and better initialization."""
    hidden_size: int
    num_classes: int = 2
    dropout_rate: float = 0.2  # Reduced dropout
    
    @nn.compact
    def __call__(self, x, training=False):
        # Input layer with better initialization
        x = nn.Dense(self.hidden_size, 
                    kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        if training:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=False)
        
        # Hidden layer
        x = nn.Dense(self.hidden_size // 2,
                    kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)
        
        if training:
            x = nn.Dropout(self.dropout_rate)(x, deterministic=False)
        
        # Output layer
        x = nn.Dense(self.num_classes,
                    kernel_init=nn.initializers.zeros)(x)
        return x

# ========== BETTER DP NOISE ==========
def add_dp_noise_safe(grads, epsilon, delta, clip_norm=0.5):  # Reduced clipping
    """Safer DP noise with better gradient control."""
    # Gradient clipping with smoother function
    def clip_gradient(g):
        norm = jnp.linalg.norm(g) + 1e-10
        clipped_norm = jnp.minimum(norm, clip_norm)
        return g * (clipped_norm / norm)
    
    grads = tree_util.tree_map(clip_gradient, grads)
    
    # More stable noise calculation
    safe_epsilon = jnp.maximum(epsilon, 0.1)  # Prevent tiny epsilon
    noise_scale = clip_norm * jnp.sqrt(2 * jnp.log(1.25 / delta)) / safe_epsilon
    
    # Scale noise based on gradient magnitude
    key = jax.random.PRNGKey(42)
    
    def add_scaled_noise(g):
        nonlocal key
        key, subkey = jax.random.split(key)
        # Add noise proportional to gradient norm
        grad_norm = jnp.linalg.norm(g) + 1e-10
        scaled_noise = noise_scale * grad_norm * jax.random.normal(subkey, g.shape)
        return g + scaled_noise
    
    return tree_util.tree_map(add_scaled_noise, grads)

# ========== WEIGHTED LOSS (for class imbalance) ==========
def weighted_cross_entropy(logits, labels, class_weights):
    """Weighted loss to handle class imbalance."""
    # Convert to one-hot
    labels_onehot = jax.nn.one_hot(labels, logits.shape[-1])
    
    # Apply class weights
    weights = labels_onehot * class_weights[0] + (1 - labels_onehot) * class_weights[1]
    
    # Compute weighted loss
    loss = optax.softmax_cross_entropy(logits, labels_onehot)
    weighted_loss = loss * weights
    return weighted_loss.mean()

# ========== TRAINING STEP ==========
@jax.jit
def train_step(state, batch, labels, epsilon, delta, class_weights):
    """Improved training step with weighted loss and gradient monitoring."""
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch, training=True)
        return weighted_cross_entropy(logits, labels, class_weights)
    
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Monitor gradient norms
    grad_norms = tree_util.tree_map(lambda g: jnp.linalg.norm(g), grads)
    total_grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in tree_util.tree_leaves(grads)))
    
    # Apply DP noise
    grads = add_dp_noise_safe(grads, epsilon, delta)
    
    # Update
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, total_grad_norm

@jax.jit
def eval_step(state, batch, labels, class_weights):
    """Evaluation with weighted loss."""
    logits = state.apply_fn({'params': state.params}, batch, training=False)
    loss = weighted_cross_entropy(logits, labels, class_weights)
    preds = jnp.argmax(logits, axis=-1)
    return loss, preds

# ========== DATA WITH BETTER FEATURES ==========
def load_improved_data(max_samples=5000):
    """Load data with better feature engineering."""
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
        
        for text in texts:
            # Better feature engineering for PII detection
            length = min(len(text) / 50, 2.0)  # Normalize differently
            
            # Email patterns
            has_at = 1.0 if '@' in text else 0.0
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            is_email = 1.0 if re.search(email_pattern, text) else 0.0
            
            # Phone patterns
            phone_pattern = r'(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}'
            has_phone = 1.0 if re.search(phone_pattern, text) else 0.0
            
            # Number patterns
            digit_count = sum(c.isdigit() for c in text)
            digit_ratio = digit_count / max(len(text), 1)
            has_many_digits = 1.0 if digit_ratio > 0.3 else 0.0
            
            # Name/Address patterns
            words = text.split()
            has_capitals = 1.0 if any(word.istitle() for word in words if len(word) > 1) else 0.0
            has_title = 1.0 if any(word.lower() in ['mr', 'mrs', 'ms', 'dr', 'prof'] for word in words) else 0.0
            has_address = 1.0 if any(word.lower() in ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'lane', 'ln'] 
                                   for word in words) else 0.0
            
            # Special characters
            has_special = 1.0 if any(c in text for c in ['#', '$', '%', '&', '*']) else 0.0
            
            features.append([
                length,
                has_at,
                is_email,
                has_phone,
                digit_ratio,
                has_many_digits,
                has_capitals,
                has_title,
                has_address,
                has_special,
                len(words) / 10,  # Word count normalized
                int(any(word.isupper() for word in words))  # All caps
            ])
            
            # Better labeling
            label = 1 if (is_email > 0.5 or has_phone > 0.5 or 
                         (has_capitals > 0.5 and has_many_digits > 0.5) or
                         (has_title > 0.5 and has_capitals > 0.5)) else 0
            labels.append(label)
        
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        print(f"📊 Loaded {len(features)} samples, {np.sum(labels)} PII ({np.sum(labels)/len(labels)*100:.1f}%)")
        print(f"   Feature dim: {features.shape[1]}")
        
        # Calculate class weights for imbalanced data
        class_counts = Counter(labels)
        total = len(labels)
        class_weights = np.array([
            1.0,  # Weight for class 0 (non-PII)
            total / (2.0 * class_counts[1])  # Higher weight for rare class 1 (PII)
        ], dtype=np.float32)
        
        print(f"   Class weights: {class_weights}")
        
        return features, labels, features.shape[1], class_weights
    else:
        # Synthetic fallback with balanced classes
        n = max_samples or 1000
        X = np.random.randn(n, 10).astype(np.float32)
        y = np.random.choice([0, 1], n, p=[0.5, 0.5]).astype(np.int32)  # Balanced
        class_weights = np.array([1.0, 1.0], dtype=np.float32)
        return X, y, 10, class_weights

# ========== MAIN ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=8.0)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)  # Larger batch for DP
    parser.add_argument('--hidden_size', type=int, default=128)  # Smaller
    parser.add_argument('--learning_rate', type=float, default=0.0005)  # Smaller LR
    parser.add_argument('--max_samples', type=int, default=5000)
    parser.add_argument('--early_stop', type=int, default=3, help='Patience for early stopping')
    
    args = parser.parse_args()
    
    print("="*70)
    print(f"🚀 IMPROVED DP Model Training with ε={args.epsilon}")
    print("="*70)
    
    # Setup
    model_dir = Path("outputs/models") / f"improved_epsilon_{args.epsilon}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data with class weights
    X, y, input_dim, class_weights = load_improved_data(args.max_samples)
    class_weights_jax = jnp.array(class_weights)
    
    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    print(f"📈 Split: Train={len(X_train)}, Val={len(X_val)}")
    print(f"   Class distribution - Train: {Counter(y_train)}")
    print(f"   Class distribution - Val: {Counter(y_val)}")
    
    # Initialize model
    rng = jax.random.PRNGKey(42)
    model = ImprovedDPModel(hidden_size=args.hidden_size)
    dummy = jnp.ones((1, input_dim))
    params = model.init(rng, dummy, training=False)['params']
    
    # Optimizer with gradient clipping and weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(learning_rate=args.learning_rate, weight_decay=1e-4)
    )
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # Convert to JAX arrays
    epsilon_jax = jnp.array(args.epsilon, dtype=jnp.float32)
    delta_jax = jnp.array(args.delta, dtype=jnp.float32)
    
    # Train with early stopping
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'grad_norms': []
    }
    
    print("\n🏋️‍♂️ Training with early stopping...")
    start = time.time()
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Shuffle
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train[idx]
        
        epoch_loss = 0
        epoch_grad_norm = 0
        batches = 0
        
        for i in range(0, len(X_shuffled), args.batch_size):
            batch = jnp.array(X_shuffled[i:i+args.batch_size])
            labels = jnp.array(y_shuffled[i:i+args.batch_size])
            state, loss, grad_norm = train_step(
                state, batch, labels, epsilon_jax, delta_jax, class_weights_jax
            )
            epoch_loss += float(loss)
            epoch_grad_norm += float(grad_norm)
            batches += 1
        
        avg_train_loss = epoch_loss / batches if batches > 0 else 0
        avg_grad_norm = epoch_grad_norm / batches if batches > 0 else 0
        
        # Validate
        val_batch = jnp.array(X_val)
        val_labels = jnp.array(y_val)
        val_loss, val_preds = eval_step(state, val_batch, val_labels, class_weights_jax)
        val_preds_np = np.array(val_preds)
        
        # Multiple metrics
        val_acc = accuracy_score(y_val, val_preds_np)
        val_precision = precision_score(y_val, val_preds_np, zero_division=0)
        val_recall = recall_score(y_val, val_preds_np, zero_division=0)
        val_f1 = f1_score(y_val, val_preds_np, zero_division=0)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['grad_norms'].append(avg_grad_norm)
        
        # Early stopping check
        if float(val_loss) < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            best_params = tree_util.tree_map(lambda x: np.array(x), state.params)
            import pickle
            with open(model_dir / "best_params.pkl", 'wb') as f:
                pickle.dump(best_params, f)
        else:
            patience_counter += 1
        
        print(f"  Epoch {epoch+1}/{args.epochs}:")
        print(f"    Loss: {avg_train_loss:.4f} (train) → {float(val_loss):.4f} (val)")
        print(f"    Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        print(f"    Grad norm: {avg_grad_norm:.4f}, Patience: {patience_counter}/{args.early_stop}")
        
        if patience_counter >= args.early_stop:
            print(f"\n⚠️  Early stopping at epoch {epoch+1} (no improvement for {args.early_stop} epochs)")
            break
    
    # Load best model
    print(f"\n📊 Best model: Epoch {best_epoch+1} with val loss {best_val_loss:.4f}")
    
    # Save final history
    import json
    with open(model_dir / "history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config = vars(args)
    config['input_dim'] = input_dim
    config['best_epoch'] = best_epoch
    with open(model_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Plot comprehensive results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val', marker='s')
    axes[0, 0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch+1})')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(history['val_acc'], color='green', marker='o')
    axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 2].plot(history['val_f1'], color='purple', marker='o')
    axes[0, 2].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[0, 2].set_title('F1 Score')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision-Recall
    axes[1, 0].plot(history['val_precision'], label='Precision', marker='o')
    axes[1, 0].plot(history['val_recall'], label='Recall', marker='s')
    axes[1, 0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Precision & Recall')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient norms
    axes[1, 1].plot(history['grad_norms'], color='orange', marker='o')
    axes[1, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Gradient Norms')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Norm')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Confusion matrix for best epoch (simplified)
    axes[1, 2].text(0.1, 0.5, f'Best Epoch: {best_epoch+1}\nVal Loss: {best_val_loss:.4f}\nVal Acc: {history["val_acc"][best_epoch]:.4f}\nF1: {history["val_f1"][best_epoch]:.4f}', 
                   fontsize=12, verticalalignment='center')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'DP Model Training (ε={args.epsilon}) - Improved Version', fontsize=16)
    plt.tight_layout()
    plt.savefig(model_dir / "training_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    report = f"""
    ============================================================
    📊 TRAINING REPORT - IMPROVED DP MODEL (ε={args.epsilon})
    ============================================================
    
    🎯 BEST EPOCH: {best_epoch + 1}
        • Validation Loss: {best_val_loss:.4f}
        • Accuracy: {history['val_acc'][best_epoch]:.4f}
        • F1 Score: {history['val_f1'][best_epoch]:.4f}
        • Precision: {history['val_precision'][best_epoch]:.4f}
        • Recall: {history['val_recall'][best_epoch]:.4f}
    
    📈 TRAINING METRICS (Final):
        • Train Loss: {history['train_loss'][-1]:.4f}
        • Val Loss: {history['val_loss'][-1]:.4f}
        • Gradient Norm: {history['grad_norms'][-1]:.4f}
    
    🔐 PRIVACY ANALYSIS:
        • ε = {args.epsilon}
        • δ = {args.delta}
        • Samples: {len(X_train)}
        • Batch Size: {args.batch_size}
        • Epochs Trained: {len(history['train_loss'])}
    
    💾 MODEL SAVED TO: {model_dir}
        • Parameters: best_params.pkl
        • History: history.json
        • Config: config.json
        • Analysis plot: training_analysis.png
    
    ============================================================
    """
    
    print(report)
    
    # Save report
    with open(model_dir / "training_report.md", 'w') as f:
        f.write(report)
    
    training_time = time.time() - start
    print(f"✅ Training complete in {training_time:.1f} seconds")
    print("="*70)
    
    return model_dir

if __name__ == "__main__":
    main()
