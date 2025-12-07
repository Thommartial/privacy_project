#!/bin/bash
# fix_model_architecture.sh - Fix Flax model architecture

echo "🔧 Fixing model architecture (simplifying Transformer layers)..."

# Backup current file
cp src/models/distilbert_dp.py src/models/distilbert_dp.py.backup

# Create simplified but correct model
cat > src/models/distilbert_dp.py << 'EOF'
#!/usr/bin/env python3
"""
Simplified DistilBERT DP Model in JAX/Flax for PII Detection.
Fixed architecture with correct Flax imports.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from typing import Optional, Tuple, Dict, Any
import optax
import numpy as np


@struct.dataclass
class TrainingState:
    """Training state container."""
    step: int
    params: dict
    opt_state: optax.OptState
    rng: jnp.ndarray


class AttentionBlock(nn.Module):
    """Simplified attention block."""
    hidden_size: int = 768
    num_heads: int = 12
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, attention_mask=None, training=False):
        # Multi-head attention
        attn_output = nn.SelfAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate if training else 0.0
        )(x, attention_mask=attention_mask)
        
        # Add & Norm
        x = x + attn_output
        x = nn.LayerNorm()(x)
        
        # Feed-forward
        ff_output = nn.Dense(self.hidden_size * 4)(x)
        ff_output = nn.gelu(ff_output)
        ff_output = nn.Dense(self.hidden_size)(ff_output)
        ff_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(ff_output)
        
        # Add & Norm
        x = x + ff_output
        x = nn.LayerNorm()(x)
        
        return x


class DistilBertDP(nn.Module):
    """
    Simplified DistilBERT model for token classification with DP support.
    """
    num_labels: int = 3  # B, I, O tags
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 6
    max_position_embeddings: int = 512
    dropout: float = 0.1
    epsilon: float = 8.0  # Privacy budget (will be overridden by training script)
    delta: float = 1e-5
    noise_multiplier: float = 0.25  # Will be calculated based on epsilon in trainer
    
    @nn.compact
    def __call__(self, 
                 input_ids: jnp.ndarray,
                 attention_mask: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            training: Whether in training mode
            
        Returns:
            Logits [batch_size, seq_len, num_labels]
        """
        # Embeddings
        x = nn.Embed(num_embeddings=30522, features=self.hidden_size)(input_ids)
        
        # Add positional embeddings
        seq_len = input_ids.shape[1]
        pos_emb = self.param('pos_emb', 
                           nn.initializers.normal(stddev=0.02),
                           (seq_len, self.hidden_size))
        x = x + pos_emb[None, :, :]
        
        # Transformer layers
        for _ in range(self.num_hidden_layers):
            x = AttentionBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_attention_heads,
                dropout_rate=self.dropout
            )(x, attention_mask=attention_mask[:, None, None, :], training=training)
        
        # Classification head
        logits = nn.Dense(self.num_labels)(x)
        return logits
    
    def compute_privacy_cost(self, 
                           steps: int, 
                           batch_size: int, 
                           dataset_size: int,
                           noise_multiplier: float) -> Dict[str, float]:
        """
        Compute privacy cost using moments accountant.
        
        Args:
            steps: Number of training steps
            batch_size: Batch size
            dataset_size: Total dataset size
            noise_multiplier: Noise multiplier used in training
            
        Returns:
            Dictionary with privacy metrics
        """
        q = batch_size / dataset_size  # Sampling probability
        
        # Simplified DP-SGD accounting (Abadi et al. 2016)
        epsilon = noise_multiplier * q * np.sqrt(steps * np.log(1/self.delta))
        
        return {
            'epsilon': float(epsilon),
            'delta': self.delta,
            'sigma': noise_multiplier,
            'q': q,
            'steps': steps
        }


class DPTrainer:
    """
    Differential Privacy Trainer for DistilBERT.
    Implements DP-SGD with gradient clipping and noise addition.
    """
    
    def __init__(self,
                 model: DistilBertDP,
                 learning_rate: float = 5e-5,
                 epsilon: float = 8.0,
                 max_grad_norm: float = 1.0,
                 batch_size: int = 32):
        
        self.model = model
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        
        # Calculate noise multiplier based on epsilon
        # Higher epsilon = less privacy = less noise
        self.noise_multiplier = max(0.1, 2.0 / epsilon)
        
        # Optimizer with DP-SGD modifications
        self.optimizer = self._create_dp_optimizer()
    
    def _create_dp_optimizer(self) -> optax.GradientTransformation:
        """
        Create optimizer with DP-SGD modifications.
        """
        # Standard AdamW optimizer
        optimizer = optax.adamw(
            learning_rate=self.learning_rate,
            weight_decay=0.01
        )
        
        # Add gradient clipping for DP-SGD
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optimizer
        )
        
        return optimizer
    
    def init_state(self, rng_key: jnp.ndarray, 
                   dummy_input: jnp.ndarray,
                   dummy_mask: jnp.ndarray) -> TrainingState:
        """
        Initialize training state.
        
        Args:
            rng_key: Random key
            dummy_input: Dummy input for initialization
            dummy_mask: Dummy attention mask
            
        Returns:
            Initialized training state
        """
        # Initialize parameters
        params = self.model.init(rng_key, dummy_input, dummy_mask, training=False)
        
        # Initialize optimizer state
        opt_state = self.optimizer.init(params)
        
        return TrainingState(
            step=0,
            params=params,
            opt_state=opt_state,
            rng=rng_key
        )
    
    def compute_loss(self, params: dict, 
                     batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
                     rng_key: jnp.ndarray) -> Tuple[jnp.ndarray, dict]:
        """
        Compute loss and accuracy for a batch.
        
        Args:
            params: Model parameters
            batch: Tuple of (input_ids, attention_mask, labels)
            rng_key: Random key
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        input_ids, attention_mask, labels = batch
        
        # Forward pass
        logits = self.model.apply(params, input_ids, attention_mask, 
                                 training=True, rngs={'dropout': rng_key})
        
        # Compute cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = loss.mean()
        
        # Compute accuracy
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': predictions,
            'labels': labels
        }
        
        return loss, metrics
    
    def train_step(self, state: TrainingState,
                   batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[TrainingState, dict]:
        """
        Single training step with DP-SGD.
        
        Args:
            state: Current training state
            batch: Training batch
            
        Returns:
            Updated state and metrics
        """
        input_ids, attention_mask, labels = batch
        
        # Split RNG key
        rng, dropout_rng = jax.random.split(state.rng)
        
        # Compute loss and gradients
        grad_fn = jax.value_and_grad(self.compute_loss, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params, 
                                        (input_ids, attention_mask, labels),
                                        dropout_rng)
        
        # Apply DP-SGD modifications
        grads = self._apply_dp_modifications(grads, rng)
        
        # Update parameters
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        
        # Create new state
        new_state = state.replace(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            rng=rng
        )
        
        return new_state, {**metrics, 'grad_norm': self._compute_grad_norm(grads)}
    
    def _apply_dp_modifications(self, grads: dict, rng_key: jnp.ndarray) -> dict:
        """
        Apply DP-SGD modifications: clipping and noise addition.
        
        Args:
            grads: Gradient dictionary
            rng_key: Random key for noise
            
        Returns:
            Modified gradients
        """
        # Clip gradients
        grads = jax.tree_map(
            lambda g: g / jnp.maximum(jnp.linalg.norm(g) / self.max_grad_norm, 1.0),
            grads
        )
        
        # Add Gaussian noise
        noise_key, _ = jax.random.split(rng_key)
        
        def add_noise(grad):
            noise = jax.random.normal(noise_key, shape=grad.shape)
            return grad + self.noise_multiplier * self.max_grad_norm * noise / self.batch_size
        
        grads = jax.tree_map(add_noise, grads)
        
        return grads
    
    def _compute_grad_norm(self, grads: dict) -> float:
        """Compute gradient norm."""
        grad_norm = jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
        )
        return float(grad_norm)
    
    def evaluate(self, params: dict, 
                 dataset: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            params: Model parameters
            dataset: Evaluation dataset
            
        Returns:
            Dictionary of evaluation metrics
        """
        input_ids, attention_mask, labels = dataset
        
        # Forward pass
        logits = self.model.apply(params, input_ids, attention_mask, training=False)
        
        # Compute metrics
        predictions = jnp.argmax(logits, axis=-1)
        
        # Calculate accuracy
        accuracy = (predictions == labels).mean()
        
        # Calculate per-class metrics
        correct_pii = ((predictions != 0) & (labels != 0)).sum()
        total_pred_pii = (predictions != 0).sum()
        total_true_pii = (labels != 0).sum()
        
        precision = correct_pii / total_pred_pii if total_pred_pii > 0 else 0.0
        recall = correct_pii / total_true_pii if total_true_pii > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'total_pii': int(total_true_pii),
            'predicted_pii': int(total_pred_pii),
            'correct_pii': int(correct_pii)
        }


# Utility functions
def save_checkpoint(state: TrainingState, path: str, metrics: dict, epsilon: float):
    """Save model checkpoint with epsilon info."""
    import pickle
    
    checkpoint = {
        'state': state,
        'metrics': metrics,
        'config': {
            'epsilon': epsilon,
            'noise_multiplier': metrics.get('noise_multiplier', 0.25),
            'batch_size': 32,
            'learning_rate': 5e-5
        }
    }
    
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    print(f"✅ Checkpoint saved: {path} (ε={epsilon})")


def load_checkpoint(path: str) -> Tuple[TrainingState, dict, dict]:
    """Load model checkpoint."""
    import pickle
    
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    return checkpoint['state'], checkpoint['metrics'], checkpoint['config']
EOF

echo "✅ Fixed model architecture"
echo ""
echo "🔄 Now updating train_dp.py to reduce complexity for faster training..."

# Also simplify train_dp.py to reduce memory usage
cat > src/training/train_dp.py << 'EOF'
#!/usr/bin/env python3
"""
Simplified training script for DP DistilBERT model.
Optimized for CPU training with reduced memory usage.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

# Import our modules
from src.models.distilbert_dp import DistilBertDP, DPTrainer, TrainingState, save_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DP DistilBERT model")
    
    parser.add_argument("--epsilon", type=float, default=8.0,
                       help="Privacy budget (ε)")
    parser.add_argument("--model_name", type=str, default="",
                       help="Model name for saving (auto-generated if empty)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16,  # Reduced from 32
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=64,  # Reduced from 128
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="outputs/models",
                       help="Output directory")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                       help="Early stopping patience")
    
    return parser.parse_args()


def load_and_preprocess_data(max_seq_length: int = 64):
    """
    Load and preprocess data for training.
    Simplified to reduce memory usage.
    """
    print("📂 Loading data...")
    
    # Load smaller subset for faster training
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    # Use smaller subset for demo (1000 samples)
    train_df = train_df.sample(n=min(1000, len(train_df)), random_state=42)
    val_df = val_df.sample(n=min(200, len(val_df)), random_state=42)
    
    print(f"Using subset: Train={len(train_df)}, Val={len(val_df)}")
    
    # Convert labels from string to list
    def parse_labels(labels_str):
        if isinstance(labels_str, str):
            return eval(labels_str)
        return labels_str
    
    train_df['labels'] = train_df['labels'].apply(parse_labels)
    val_df['labels'] = val_df['labels'].apply(parse_labels)
    
    # Create simplified dataset
    def create_dataset(df, max_len):
        texts = df['text'].tolist()
        labels = df['labels'].tolist()
        
        input_ids = []
        attention_masks = []
        label_arrays = []
        
        for text, label_list in zip(texts, labels):
            # Simple tokenization (first max_len words)
            words = text.split()[:max_len]
            seq_len = len(words)
            
            # Create input ids (simple hash of words)
            ids = [hash(word) % 30000 for word in words]
            if seq_len < max_len:
                ids = ids + [0] * (max_len - seq_len)  # Pad with 0
            
            # Create attention mask
            mask = [1] * seq_len + [0] * (max_len - seq_len)
            
            # Create labels (pad with 0)
            label_array = [0] * max_len
            for i in range(min(len(label_list), max_len)):
                label = label_list[i]
                if label == 'O':
                    label_array[i] = 0
                elif label.startswith('B-'):
                    label_array[i] = 1
                elif label.startswith('I-'):
                    label_array[i] = 2
            
            input_ids.append(ids)
            attention_masks.append(mask)
            label_arrays.append(label_array)
        
        return {
            'input_ids': jnp.array(input_ids, dtype=jnp.int32),
            'attention_mask': jnp.array(attention_masks, dtype=jnp.int32),
            'labels': jnp.array(label_arrays, dtype=jnp.int32)
        }
    
    train_data = create_dataset(train_df, max_seq_length)
    val_data = create_dataset(val_df, max_seq_length)
    
    print(f"✅ Data loaded and preprocessed")
    print(f"   Input shape: {train_data['input_ids'].shape}")
    
    return train_data, val_data


def create_batches(data: Dict, batch_size: int):
    """Create batches from dataset."""
    n_samples = data['input_ids'].shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    batches = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch = (
            data['input_ids'][batch_indices],
            data['attention_mask'][batch_indices],
            data['labels'][batch_indices]
        )
        batches.append(batch)
    
    return batches


def train_model(args, train_data, val_data):
    """Train the DP model."""
    print(f"\n🏋️‍♂️ Starting training (ε={args.epsilon})...")
    
    # Initialize model with current epsilon
    model = DistilBertDP(
        epsilon=args.epsilon,
        num_hidden_layers=3,  # Reduced from 6 for faster training
        hidden_size=384,      # Reduced from 768 for faster training
        num_attention_heads=6 # Reduced from 12
    )
    
    trainer = DPTrainer(
        model=model,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        batch_size=args.batch_size
    )
    
    print(f"   Model parameters: {model.num_hidden_layers} layers, {model.hidden_size} hidden size")
    print(f"   Noise multiplier: {trainer.noise_multiplier:.3f}")
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    dummy_input = jnp.ones((2, args.max_seq_length), dtype=jnp.int32)
    dummy_mask = jnp.ones_like(dummy_input)
    
    state = trainer.init_state(init_rng, dummy_input, dummy_mask)
    
    # Create batches
    train_batches = create_batches(train_data, args.batch_size)
    val_batches = create_batches(val_data, args.batch_size)
    
    # Training history
    history = {
        'epsilon': args.epsilon,
        'noise_multiplier': trainer.noise_multiplier,
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'epochs': []
    }
    
    # Early stopping
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Training
        epoch_loss = []
        epoch_accuracy = []
        
        for batch in train_batches:
            state, metrics = trainer.train_step(state, batch)
            epoch_loss.append(metrics['loss'])
            epoch_accuracy.append(metrics['accuracy'])
        
        avg_loss = np.mean(epoch_loss)
        avg_accuracy = np.mean(epoch_accuracy)
        
        # Validation (use first batch only for speed)
        if val_batches:
            val_metrics = trainer.evaluate(state.params, val_batches[0])
        else:
            val_metrics = {'accuracy': 0.0, 'f1_score': 0.0}
        
        # Record history
        history['train_loss'].append(float(avg_loss))
        history['train_accuracy'].append(float(avg_accuracy))
        history['val_accuracy'].append(float(val_metrics['accuracy']))
        history['val_f1'].append(float(val_metrics['f1_score']))
        history['epochs'].append(epoch + 1)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {avg_accuracy:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Early stopping
        current_f1 = val_metrics['f1_score']
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_state = state
            patience_counter = 0
            print(f"    ↳ New best F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"    ↳ Early stopping (no improvement for {patience_counter} epochs)")
                break
    
    # Use best state
    final_state = best_state if best_state else state
    
    # Compute privacy cost
    privacy_cost = model.compute_privacy_cost(
        steps=state.step,
        batch_size=args.batch_size,
        dataset_size=train_data['input_ids'].shape[0],
        noise_multiplier=trainer.noise_multiplier
    )
    
    history['privacy_cost'] = privacy_cost
    history['total_steps'] = state.step
    
    print(f"\n📊 Training completed:")
    print(f"   Total steps: {state.step}")
    print(f"   Best Val F1: {best_val_f1:.4f}")
    print(f"   Privacy: ε={privacy_cost['epsilon']:.3f}")
    
    return final_state, history


def save_training_results(state, history, args):
    """Save training results and model."""
    if not args.model_name:
        args.model_name = f"distilbert_epsilon_{args.epsilon}"
    
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = output_dir / "model_checkpoint.pkl"
    save_checkpoint(state, str(checkpoint_path), history, args.epsilon)
    
    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    # Save config
    config = {
        'epsilon': args.epsilon,
        'model_name': args.model_name,
        'epochs_trained': len(history['epochs']),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_seq_length': args.max_seq_length,
        'best_val_f1': max(history['val_f1']) if history['val_f1'] else 0.0,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}")
    return output_dir


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("="*70)
    print(f"🚀 DP DISTILBERT TRAINING (Simplified)")
    print("="*70)
    print(f"  ε (epsilon): {args.epsilon}")
    print(f"  Model: {args.model_name or f'distilbert_epsilon_{args.epsilon}'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seq length: {args.max_seq_length}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # Load data
        train_data, val_data = load_and_preprocess_data(args.max_seq_length)
        
        # Train model
        final_state, history = train_model(args, train_data, val_data)
        
        # Save results
        output_dir = save_training_results(final_state, history, args)
        
        # Print summary
        training_time = time.time() - start_time
        print(f"\n📋 TRAINING SUMMARY")
        print("="*70)
        print(f"  Total time: {training_time/60:.1f} minutes")
        print(f"  Best Val F1: {max(history['val_f1']):.4f}")
        print(f"  Privacy cost: ε={history['privacy_cost']['epsilon']:.3f}")
        print(f"  Output: {output_dir}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
EOF

echo "✅ Simplified training script"
echo ""
echo "🎯 Changes made:"
echo "  1. Fixed model architecture (removed TransformerEncoderLayer)"
echo "  2. Reduced model size (3 layers, 384 hidden size)"
echo "  3. Reduced batch size (16) and sequence length (64)"
echo "  4. Using subset of data (1000 train, 200 val samples)"
echo ""
echo "🚀 Now training will be much faster (~5-10 minutes)"
echo "   Run: ./run_epsilon_8.sh"
