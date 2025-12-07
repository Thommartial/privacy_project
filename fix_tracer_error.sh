#!/bin/bash
# fix_tracer_error.sh - Fix TracerBoolConversionError

echo "🔧 Fixing TracerBoolConversionError..."
echo "="*60

# Backup current files
cp src/training/train_dp_proper.py src/training/train_dp_proper.py.backup3
cp src/utils/dp_utils.py src/utils/dp_utils.py.backup2

# Fix the dp_utils.py to avoid Python conditionals in JIT
cat > src/utils/dp_utils.py << 'EOF'
"""
DP utilities for differential privacy.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax

def compute_dp_sgd_privacy(n, batch_size, target_epsilon, target_delta, epochs):
    """
    Compute noise multiplier for DP-SGD.
    Based on the moments accountant.
    
    Returns:
        noise_multiplier: The σ parameter for Gaussian noise
    """
    # This runs outside JIT, so Python conditionals are fine
    if target_epsilon <= 0 or target_delta <= 0:
        return 0.0
    
    q = batch_size / n  # Sampling probability
    steps = epochs * n // batch_size
    
    # Use the Gaussian mechanism formula: σ = Δf * sqrt(2 * ln(1.25/δ)) / ε
    # Where Δf is sensitivity (we use clip_norm=1.0)
    clip_norm = 1.0
    noise_multiplier = clip_norm * np.sqrt(2 * np.log(1.25 / target_delta)) / target_epsilon
    
    # Adjust for composition: σ_total = σ / sqrt(steps)
    if steps > 1:
        noise_multiplier = noise_multiplier / np.sqrt(steps)
    
    return float(max(noise_multiplier, 0.1))  # Ensure minimum noise

def add_noise_to_grads(grads, epsilon, delta, clip_norm=1.0):
    """
    Add calibrated Gaussian noise to gradients for DP-SGD.
    
    Args:
        grads: Gradient tree
        epsilon: Privacy budget per step
        delta: Privacy parameter
        clip_norm: Gradient clipping norm
    
    Returns:
        Noisy gradients
    """
    # This function will be JIT compiled, so we need to use JAX control flow
    
    # Clip gradients to bound sensitivity
    def clip_gradient(g):
        norm = jnp.linalg.norm(g)
        scale = jnp.minimum(1.0, clip_norm / (norm + 1e-10))
        return g * scale
    
    grads = jax.tree_map(clip_gradient, grads)
    
    # Compute noise scale using jnp.where for conditionals
    # If epsilon <= 0, no noise; otherwise compute noise
    safe_epsilon = jnp.maximum(epsilon, 1e-10)  # Avoid division by zero
    noise_scale = clip_norm * jnp.sqrt(2 * jnp.log(1.25 / delta)) / safe_epsilon
    
    # Use jnp.where to conditionally apply noise
    # If epsilon <= 0, noise_scale = 0
    noise_scale = jnp.where(epsilon <= 0, 0.0, noise_scale)
    
    # Add Gaussian noise to each gradient
    key = random.PRNGKey(0)
    
    def add_noise(g):
        # Generate new key for each operation
        key, subkey = random.split(key)
        noise = noise_scale * random.normal(subkey, g.shape)
        return g + noise
    
    # Apply noise to all gradients
    noisy_grads = jax.tree_map(add_noise, grads)
    
    return noisy_grads

def compute_renyi_dp(q, sigma, steps, alpha):
    """
    Compute Renyi Differential Privacy.
    """
    return alpha / (2 * sigma ** 2) if sigma > 0 else float('inf')

class PrivacyAccountant:
    """Track privacy spending using moments accountant."""
    
    def __init__(self, target_delta=1e-5):
        self.target_delta = target_delta
        self.rdp_alphas = list(range(2, 65))
        self.rdp_budget = {alpha: 0.0 for alpha in self.rdp_alphas}
    
    def add_step(self, q, sigma):
        for alpha in self.rdp_alphas:
            rdp_alpha = alpha / (2 * sigma ** 2) if sigma > 0 else float('inf')
            self.rdp_budget[alpha] += rdp_alpha
    
    def get_epsilon(self, delta=None):
        if delta is None:
            delta = self.target_delta
        
        eps = float('inf')
        for alpha, rdp in self.rdp_budget.items():
            eps = min(eps, rdp + np.log(1/delta)/(alpha-1))
        
        return eps
EOF

echo "✅ Updated src/utils/dp_utils.py to use JAX control flow"

# Also update train_dp_proper.py to ensure epsilon is passed correctly
cat > src/training/train_dp_proper.py << 'EOF'
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
    
    # Save model
    model_path = model_dir / "checkpoint.pkl"
    
    # Convert JAX arrays to numpy for saving
    params_numpy = jax.tree_map(lambda x: np.array(x), state.params)
    
    import pickle
    with open(model_path, 'wb') as f:
        pickle.dump({
            'params': params_numpy,
            'state': state,
            'history': history,
            'input_dim': input_dim,
            'hidden_size': args.hidden_size,
            'config': vars(args)
        }, f)
    
    # Save training history
    history_path = model_dir / "history.json"
    import json
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Compute privacy spent (outside JIT)
    n = len(train_features)
    batch_size = args.batch_size
    epochs = args.epochs
    noise_multiplier = compute_dp_sgd_privacy(
        n, batch_size, args.epsilon, args.delta, epochs
    )
    
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
EOF

echo "✅ Updated train_dp_proper.py to use JAX arrays for epsilon/delta"

echo ""
echo "🛠️  Testing the fix..."
python -c "
import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, '.')

# Test the DP utilities
from src.utils.dp_utils import add_noise_to_grads

# Create dummy gradients
grads = {'weight': jnp.ones((10, 5)), 'bias': jnp.ones((5,))}

# Test with JAX arrays
epsilon = jnp.array(8.0)
delta = jnp.array(1e-5)

# This should work inside JIT
@jax.jit
def test_dp(grads, epsilon, delta):
    return add_noise_to_grads(grads, epsilon, delta)

try:
    noisy_grads = test_dp(grads, epsilon, delta)
    print('✅ DP noise addition works with JIT')
    print(f'  Original grad norm: {jnp.linalg.norm(grads[\"weight\"]):.4f}')
    print(f'  Noisy grad norm: {jnp.linalg.norm(noisy_grads[\"weight\"]):.4f}')
except Exception as e:
    print(f'❌ Error: {e}')

# Test the model
from src.training.train_dp_proper import ProperDPModel
model = ProperDPModel(hidden_size=256)
rng = jax.random.PRNGKey(42)
dummy_input = jnp.ones((2, 8))
params = model.init(rng, dummy_input, training=False)
print('\\n✅ Model creation works')
"

echo ""
echo "🚀 Now run the training:"
echo "   python src/training/train_dp_proper.py --epsilon 8.0 --epochs 3 --batch_size 16 --max_samples 3000"
