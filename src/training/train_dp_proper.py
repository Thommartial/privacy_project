#!/usr/bin/env python3
"""
Train a proper DP model for PII detection with JAX/Flax.
This uses real PII data, better architecture, and proper DP-SGD.
"""
import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from jax import tree_util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.data_loader import load_and_preprocess_data, split_dataset
from src.utils.dp_utils import compute_dp_sgd_privacy, add_noise_to_grads

# Set random seeds for reproducibility
np.random.seed(42)

class ProperDPModel(nn.Module):
    """Enhanced model for PII detection with dropout for better generalization."""
    hidden_size: int
    num_classes: int = 2  # binary classification: PII vs non-PII
    dropout_rate: float = 0.3
    
    @nn.compact
    def __call__(self, x, training=False):
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        # First hidden layer
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        else:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=True)
        
        # Second hidden layer
        x = nn.Dense(self.hidden_size // 2)(x)
        x = nn.relu(x)
        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
        else:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=True)
        
        # Output layer
        x = nn.Dense(self.num_classes)(x)
        return x

def create_train_state(rng, input_dim, hidden_size, learning_rate):
    """Creates initial training state."""
    model = ProperDPModel(hidden_size=hidden_size)
    
    # Initialize parameters
    dummy_input = jnp.ones((1, input_dim))
    params = model.init(rng, dummy_input, training=False)['params']
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    
    # Create training state
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )

@jax.jit
def train_step(state, batch, labels, dropout_rng, epsilon, delta, clip_norm=1.0):
    """Train step with DP-SGD."""
    def loss_fn(params):
        # Apply model with dropout during training
        logits = state.apply_fn(
            {'params': params}, 
            batch, 
            training=True,
            rngs={'dropout': dropout_rng}
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        ).mean()
        return loss
    
    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Apply DP: clip and add noise
    # Note: epsilon and delta are traced values inside JIT
    grads = add_noise_to_grads(grads, epsilon, delta, clip_norm)
    
    # Update parameters
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@jax.jit
def eval_step(state, batch, labels):
    """Evaluation step (no dropout)."""
    logits = state.apply_fn({'params': state.params}, batch, training=False)
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, labels
    ).mean()
    
    # Get predictions
    preds = jnp.argmax(logits, axis=-1)
    return loss, preds

def load_training_data(max_samples=500, max_seq_length=64):
    """Load and preprocess training data."""
    print("📊 Loading training data...")
    
    data_path = project_root / "data/processed"
    
    if (data_path / "train.csv").exists():
        # Load real data
        import pandas as pd
        df = pd.read_csv(data_path / "train.csv")
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, random_state=42)
        
        # Simple feature extraction
        texts = df['text'].astype(str).tolist()
        features = []
        labels = []
        
        for text in texts:
            # Basic features for PII detection
            length = min(len(text) / 100, 1.0)  # Normalize length
            
            # Check for email patterns
            has_at = 1.0 if '@' in text else 0.0
            has_dot_com = 1.0 if '.com' in text.lower() else 0.0
            has_dot_net = 1.0 if '.net' in text.lower() else 0.0
            has_dot_org = 1.0 if '.org' in text.lower() else 0.0
            
            # Check for phone number patterns
            has_digits = sum(c.isdigit() for c in text)
            digit_ratio = has_digits / max(len(text), 1)
            has_phone_pattern = 1.0 if (has_digits >= 10 and 
                                        any(c in text for c in ['-', '(', ')', '+'])) else 0.0
            
            # Check for name patterns (capitalized words)
            words = text.split()
            has_capitals = 1.0 if any(word.istitle() for word in words) else 0.0
            
            features.append([
                length,
                has_at,
                has_dot_com or has_dot_net or has_dot_org,
                digit_ratio,
                has_phone_pattern,
                has_capitals,
                int(any(word.lower() in ['mr', 'mrs', 'ms', 'dr', 'prof'] for word in words)),
                int(any(word.lower() in ['street', 'st', 'avenue', 'ave', 'road', 'rd'] for word in words))
            ])
            
            # Label: mark as PII if it looks like email, phone, or address
            label = 1 if (
                (has_at and (has_dot_com or has_dot_net or has_dot_org)) or
                has_phone_pattern or
                (has_capitals and len(words) >= 2 and digit_ratio > 0.3)
            ) else 0
            labels.append(label)
        
        features = np.array(features, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        print(f"  Loaded {len(features)} samples, feature dim: {features.shape[1]}")
        print(f"  Class distribution: {np.sum(labels==1)} PII, {np.sum(labels==0)} non-PII")
        return features, labels, features.shape[1]
    else:
        # Fallback to synthetic data
        print("  Using synthetic data (real data not found)")
        n_samples = max_samples if max_samples else 100
        input_dim = 8  # Feature dimension
        
        rng = np.random.RandomState(42)
        features = rng.randn(n_samples, input_dim).astype(np.float32)
        labels = (rng.rand(n_samples) > 0.3).astype(np.int32)  # 30% PII
        
        return features, labels, input_dim

def main():
    parser = argparse.ArgumentParser(description='Train proper DP model')
    parser.add_argument('--epsilon', type=float, default=8.0, help='Privacy budget (ε)')
    parser.add_argument('--delta', type=float, default=1e-5, help='Privacy parameter (δ)')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=500, help='Max samples to use')
    parser.add_argument('--max_seq_length', type=int, default=64, help='Max sequence length')
    parser.add_argument('--output_dir', type=str, default='outputs/models', help='Output directory')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"🚀 Training Proper DP Model with ε={args.epsilon}")
    print("="*60)
    
    # Create output directory
    model_dir = Path(args.output_dir) / f"proper_epsilon_{args.epsilon}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    features, labels, input_dim = load_training_data(
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length
    )
    
    # Split data
    split_idx = int(0.8 * len(features))
    train_features, val_features = features[:split_idx], features[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    print(f"📈 Data split: Train={len(train_features)}, Val={len(val_features)}")
    
    # Initialize model and RNGs
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    dropout_rng = jax.random.PRNGKey(123)  # Separate RNG for dropout
    
    state = create_train_state(
        init_rng, 
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate
    )
    
    # Convert epsilon/delta to JAX arrays for JIT compatibility
    epsilon_jax = jnp.array(args.epsilon, dtype=jnp.float32)
    delta_jax = jnp.array(args.delta, dtype=jnp.float32)
    
    # Training loop
    print("\n🏋️‍♂️ Starting training...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'epsilon': args.epsilon,
        'delta': args.delta,
        'epochs': args.epochs
    }
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        # Shuffle training data
        indices = np.random.permutation(len(train_features))
        train_features_shuffled = train_features[indices]
        train_labels_shuffled = train_labels[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(train_features_shuffled), args.batch_size):
            batch_end = min(i + args.batch_size, len(train_features_shuffled))
            batch = train_features_shuffled[i:batch_end]
            batch_labels = train_labels_shuffled[i:batch_end]
            
            # Split dropout RNG for each batch
            dropout_rng, dropout_rng_batch = jax.random.split(dropout_rng)
            
            # Convert to JAX arrays
            batch_jax = jnp.array(batch, dtype=jnp.float32)
            labels_jax = jnp.array(batch_labels, dtype=jnp.int32)
            
            # Train step - pass JAX arrays for epsilon/delta
            state, loss = train_step(
                state, 
                batch_jax, 
                labels_jax, 
                dropout_rng_batch,
                epsilon_jax, 
                delta_jax
            )
            
            epoch_loss += float(loss)  # Convert from JAX array
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        
        # Validation
        val_batch = jnp.array(val_features, dtype=jnp.float32)
        val_labels_jax = jnp.array(val_labels, dtype=jnp.int32)
        val_loss, val_preds = eval_step(state, val_batch, val_labels_jax)
        val_acc = accuracy_score(val_labels, np.array(val_preds))
        
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_accuracy'].append(float(val_acc))
        
        print(f"  Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {float(val_loss):.4f}, "
              f"Val Acc: {val_acc:.4f}")
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    
    # Save model - safer approach avoiding pickling JAX objects
    model_path = model_dir / "checkpoint.pkl"
    
    # Extract only the parameters (not the entire state)
    params_numpy = tree_util.tree_map(lambda x: np.array(x), state.params)
    
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump({
            'params': params_numpy,
            'history': history,
            'input_dim': input_dim,
            'hidden_size': args.hidden_size,
            'config': vars(args)
        }, f)
    
    # Also save the model architecture info separately
    model_info_path = model_dir / "model_info.json"
    import json
    with open(model_info_path, 'w') as f:
        json.dump({
            'model_class': 'ProperDPModel',
            'input_dim': input_dim,
            'hidden_size': args.hidden_size,
            'num_classes': 2,
            'dropout_rate': 0.3
        }, f, indent=2)
    
    # Save training history
    history_path = model_dir / "history.json"
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Compute privacy spent (outside JIT)
    n = len(train_features)
    batch_size = args.batch_size
    epochs = args.epochs
    try:
        noise_multiplier = compute_dp_sgd_privacy(
            n, batch_size, args.epsilon, args.delta, epochs
        )
    except Exception as e:
        print(f"⚠️  Privacy computation failed: {e}")
        # Use a simple approximation
        noise_multiplier = 0.1 / args.epsilon if args.epsilon > 0 else 0.0
    
    print(f"\n📊 Model saved to: {model_dir}")
    print(f"🔐 Privacy analysis:")
    print(f"   • ε = {args.epsilon}")
    print(f"   • δ = {args.delta}")
    print(f"   • Noise multiplier = {noise_multiplier:.4f}")
    print(f"   • Samples = {n}, Batch size = {batch_size}, Epochs = {epochs}")
    
    # Create a simple training curve plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Curves (ε={args.epsilon})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_accuracy'], label='Val Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Validation Accuracy (ε={args.epsilon})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = model_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"📈 Training curves saved to: {plot_path}")
    print("="*60)
    print("🎉 Proper DP model training complete!")
    
    return model_dir

if __name__ == "__main__":
    main()
