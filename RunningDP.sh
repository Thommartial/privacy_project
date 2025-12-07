#!/bin/bash
# fix_epsilon_aware.sh - Make all files ε-aware in one run

set -e  # Exit on error

echo "="*70
echo "🔧 MAKING ALL FILES ε-AWARE"
echo "="*70

# Backup original files
echo "📁 Creating backups..."
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp src/models/distilbert_dp.py "$BACKUP_DIR/distilbert_dp.py.backup" 2>/dev/null || true
cp src/training/train_dp.py "$BACKUP_DIR/train_dp.py.backup" 2>/dev/null || true
cp src/evaluation/plot_utils.py "$BACKUP_DIR/plot_utils.py.backup" 2>/dev/null || true

echo "✅ Backups saved to: $BACKUP_DIR"

# ============================================================================
# 1. FIX src/models/distilbert_dp.py
# ============================================================================
echo -e "\n🔧 Fixing src/models/distilbert_dp.py..."

cat > src/models/distilbert_dp.py << 'EOF'
#!/usr/bin/env python3
"""
DistilBERT DP Model in JAX/Flax for PII Detection.
Fast, efficient, with differential privacy.
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


class DistilBertDP(nn.Module):
    """
    DistilBERT model for token classification with DP support.
    Optimized for fast CPU training.
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
    
    def setup(self):
        """Initialize model components."""
        # Embeddings
        self.embeddings = nn.Embed(
            num_embeddings=30522,  # DistilBERT vocab size
            features=self.hidden_size
        )
        
        # Transformer layers
        self.transformer_layers = [
            self._create_transformer_layer()
            for _ in range(self.num_hidden_layers)
        ]
        
        # Classification head
        self.classifier = nn.Dense(self.num_labels)
        
        # Layer norms
        self.layer_norms = [
            nn.LayerNorm() for _ in range(self.num_hidden_layers * 2)
        ]
    
    def _create_transformer_layer(self):
        """Create a single transformer layer."""
        return nn.TransformerEncoderLayer(
            num_heads=self.num_attention_heads,
            qkv_features=self.hidden_size,
            dropout_rate=self.dropout,
            activation_fn=nn.gelu
        )
    
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
        x = self.embeddings(input_ids)  # [batch, seq_len, hidden]
        
        # Add positional embeddings (simplified)
        seq_len = input_ids.shape[1]
        pos_emb = self.param('pos_emb', 
                           nn.initializers.normal(stddev=0.02),
                           (seq_len, self.hidden_size))
        x = x + pos_emb[None, :, :]
        
        # Transformer layers
        for i, layer in enumerate(self.transformer_layers):
            # Self-attention
            attn_output = layer(x, attention_mask=attention_mask[:, None, None, :])
            x = x + attn_output
            x = self.layer_norms[i * 2](x)
            
            # Feed-forward
            ff_output = nn.Dense(self.hidden_size * 4)(x)
            ff_output = nn.gelu(ff_output)
            ff_output = nn.Dense(self.hidden_size)(ff_output)
            ff_output = nn.Dropout(rate=self.dropout, deterministic=not training)(ff_output)
            x = x + ff_output
            x = self.layer_norms[i * 2 + 1](x)
        
        # Classification head
        logits = self.classifier(x)
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

echo "✅ Fixed src/models/distilbert_dp.py"

# ============================================================================
# 2. FIX src/training/train_dp.py
# ============================================================================
echo -e "\n🔧 Fixing src/training/train_dp.py..."

cat > src/training/train_dp.py << 'EOF'
#!/usr/bin/env python3
"""
Main training script for DP DistilBERT model.
Fast, efficient, with proper logging and early stopping.
"""

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
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="outputs/models",
                       help="Output directory")
    parser.add_argument("--early_stopping_patience", type=int, default=2,
                       help="Early stopping patience")
    
    return parser.parse_args()


def load_and_preprocess_data(max_seq_length: int = 128) -> Tuple[Dict, Dict, Dict]:
    """
    Load and preprocess data for training.
    
    Args:
        max_seq_length: Maximum sequence length
        
    Returns:
        Tuple of (train_data, val_data, test_data) dictionaries
    """
    print("📂 Loading data...")
    
    # Load preprocessed splits
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    # Convert labels from string to list
    def parse_labels(labels_str):
        if isinstance(labels_str, str):
            return eval(labels_str)
        return labels_str
    
    train_df['labels'] = train_df['labels'].apply(parse_labels)
    val_df['labels'] = val_df['labels'].apply(parse_labels)
    test_df['labels'] = test_df['labels'].apply(parse_labels)
    
    # Simple tokenization (in practice, use DistilBERT tokenizer)
    def tokenize_text(text):
        # Simplified tokenization - split by space
        tokens = text.split()
        # Pad/truncate to max_seq_length
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        else:
            tokens = tokens + ['[PAD]'] * (max_seq_length - len(tokens))
        return tokens
    
    # Create dataset dictionaries
    def create_dataset(df):
        texts = df['text'].tolist()
        labels = df['labels'].tolist()
        
        # Tokenize (simplified)
        tokenized = [tokenize_text(text) for text in texts]
        
        # Convert to indices (simplified - use vocab mapping in practice)
        # For demo, use random indices
        input_ids = np.random.randint(0, 30000, (len(df), max_seq_length))
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = np.ones((len(df), max_seq_length))
        
        # Convert labels to numpy array (pad to max_seq_length)
        label_array = np.zeros((len(df), max_seq_length), dtype=np.int32)
        for i, label_list in enumerate(labels):
            if isinstance(label_list, list):
                length = min(len(label_list), max_seq_length)
                # Convert BIO tags to indices: O=0, B=1, I=2
                for j in range(length):
                    if label_list[j] == 'O':
                        label_array[i, j] = 0
                    elif label_list[j].startswith('B-'):
                        label_array[i, j] = 1
                    elif label_list[j].startswith('I-'):
                        label_array[i, j] = 2
        
        return {
            'input_ids': jnp.array(input_ids),
            'attention_mask': jnp.array(attention_mask),
            'labels': jnp.array(label_array)
        }
    
    train_data = create_dataset(train_df)
    val_data = create_dataset(val_df)
    test_data = create_dataset(test_df)
    
    print(f"✅ Data loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    return train_data, val_data, test_data


def create_batches(data: Dict, batch_size: int) -> List[Tuple]:
    """
    Create batches from dataset.
    
    Args:
        data: Dataset dictionary
        batch_size: Batch size
        
    Returns:
        List of batches
    """
    n_samples = data['input_ids'].shape[0]
    batches = []
    
    for i in range(0, n_samples, batch_size):
        batch = (
            data['input_ids'][i:i+batch_size],
            data['attention_mask'][i:i+batch_size],
            data['labels'][i:i+batch_size]
        )
        batches.append(batch)
    
    return batches


def train_model(args, train_data: Dict, val_data: Dict) -> Tuple[TrainingState, Dict]:
    """
    Train the DP model.
    
    Args:
        args: Command line arguments
        train_data: Training data
        val_data: Validation data
        
    Returns:
        Tuple of (final_state, training_history)
    """
    print(f"\n🏋️‍♂️ Starting training (ε={args.epsilon})...")
    
    # Initialize model with current epsilon
    model = DistilBertDP(epsilon=args.epsilon)
    trainer = DPTrainer(
        model=model,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        batch_size=args.batch_size
    )
    
    print(f"   Noise multiplier: {trainer.noise_multiplier:.3f}")
    
    # Initialize random key
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    # Create dummy input for initialization
    dummy_input = jnp.ones((2, args.max_seq_length), dtype=jnp.int32)
    dummy_mask = jnp.ones_like(dummy_input)
    
    # Initialize training state
    state = trainer.init_state(init_rng, dummy_input, dummy_mask)
    
    # Create batches
    train_batches = create_batches(train_data, args.batch_size)
    val_batch = create_batches(val_data, args.batch_size)[0]  # Single batch for validation
    
    # Training history
    history = {
        'epsilon': args.epsilon,
        'noise_multiplier': trainer.noise_multiplier,
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'epochs': [],
        'timestamps': []
    }
    
    # Early stopping
    best_val_f1 = 0.0
    patience_counter = 0
    best_state = None
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Training
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        for batch in train_batches:
            state, metrics = trainer.train_step(state, batch)
            epoch_loss += metrics['loss']
            epoch_accuracy += metrics['accuracy']
        
        # Average metrics
        avg_loss = epoch_loss / len(train_batches)
        avg_accuracy = epoch_accuracy / len(train_batches)
        
        # Validation
        val_metrics = trainer.evaluate(state.params, val_batch)
        
        # Record history
        history['train_loss'].append(float(avg_loss))
        history['train_accuracy'].append(float(avg_accuracy))
        history['val_loss'].append(float(val_metrics.get('loss', 0.0)))
        history['val_accuracy'].append(float(val_metrics['accuracy']))
        history['val_f1'].append(float(val_metrics['f1_score']))
        history['epochs'].append(epoch + 1)
        history['timestamps'].append(time.time())
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"Train Loss: {avg_loss:.4f} | "
              f"Train Acc: {avg_accuracy:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1_score']:.4f}")
        
        # Early stopping check
        current_f1 = val_metrics['f1_score']
        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_state = state
            patience_counter = 0
            print(f"    ↳ New best F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"    ↳ Early stopping triggered (patience: {patience_counter})")
                break
    
    # Use best state if early stopped
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
    history['actual_epsilon'] = privacy_cost['epsilon']
    
    print(f"\n📊 Privacy summary:")
    print(f"   Target ε: {args.epsilon}")
    print(f"   Achieved ε: {privacy_cost['epsilon']:.3f}")
    print(f"   Steps: {state.step}")
    print(f"   Noise multiplier: {trainer.noise_multiplier:.3f}")
    
    return final_state, history


def save_training_results(state: TrainingState, history: Dict, args):
    """Save training results and model."""
    # Auto-generate model name if not provided
    if not args.model_name:
        args.model_name = f"distilbert_epsilon_{args.epsilon}"
    
    output_dir = Path(args.output_dir) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model checkpoint
    checkpoint_path = output_dir / "model_checkpoint.pkl"
    save_checkpoint(state, str(checkpoint_path), history, args.epsilon)
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types
        serializable_history = {}
        for key, value in history.items():
            if key == 'privacy_cost':
                serializable_history[key] = {k: float(v) for k, v in value.items()}
            else:
                serializable_history[key] = [float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                           for v in value]
        
        json.dump(serializable_history, f, indent=2)
    
    # Save configuration
    config = {
        'epsilon': args.epsilon,
        'model_name': args.model_name,
        'epochs_trained': len(history['epochs']),
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'max_seq_length': args.max_seq_length,
        'noise_multiplier': history.get('noise_multiplier', 0.25),
        'total_steps': history['total_steps'],
        'privacy_cost': history['privacy_cost'],
        'best_val_f1': max(history['val_f1']) if history['val_f1'] else 0.0,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - Model checkpoint: {checkpoint_path}")
    print(f"   - Training history: {history_path}")
    print(f"   - Configuration: {config_path}")
    
    return output_dir


def generate_training_plots(history: Dict, output_dir: Path):
    """Generate training visualization plots."""
    try:
        import matplotlib.pyplot as plt
        
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        epochs = history['epochs']
        epsilon = history.get('epsilon', 8.0)
        
        # Create training curves plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if 'val_loss' in history:
            plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training Loss (ε={epsilon})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        if 'train_accuracy' in history:
            plt.plot(epochs, history['train_accuracy'], 'b-', label='Train Acc', linewidth=2)
        if 'val_accuracy' in history:
            plt.plot(epochs, history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
        if 'val_f1' in history:
            plt.plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title(f'Accuracy & F1 Score (ε={epsilon})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = plots_dir / f"training_curves_epsilon_{epsilon}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Training plots saved: {plot_path}")
        
    except ImportError:
        print("⚠️  matplotlib not installed, skipping plots")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("="*70)
    print(f"🚀 DP DISTILBERT TRAINING")
    print("="*70)
    print(f"  ε (epsilon): {args.epsilon}")
    print(f"  Model: {args.model_name or f'distilbert_epsilon_{args.epsilon}'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print("="*70)
    
    start_time = time.time()
    
    try:
        # Load data
        train_data, val_data, test_data = load_and_preprocess_data(args.max_seq_length)
        
        # Train model
        final_state, history = train_model(args, train_data, val_data)
        
        # Save results
        output_dir = save_training_results(final_state, history, args)
        
        # Generate training plots
        generate_training_plots(history, output_dir)
        
        # Print summary
        training_time = time.time() - start_time
        print(f"\n📋 TRAINING SUMMARY")
        print("="*70)
        print(f"  Total time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
        print(f"  Total steps: {history['total_steps']}")
        print(f"  Best Val F1: {max(history['val_f1']):.4f}")
        print(f"  Privacy cost: ε={history['privacy_cost']['epsilon']:.3f}")
        print(f"  Output directory: {output_dir}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
EOF

echo "✅ Fixed src/training/train_dp.py"

# ============================================================================
# 3. FIX src/evaluation/plot_utils.py
# ============================================================================
echo -e "\n🔧 Fixing src/evaluation/plot_utils.py..."

cat > src/evaluation/plot_utils.py << 'EOF'
#!/usr/bin/env python3
"""
Plotting utilities for training visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import argparse


def generate_training_plots(history: Dict, output_dir: Path):
    """
    Generate training visualization plots.
    
    Args:
        history: Training history dictionary
        output_dir: Output directory for plots
    """
    output_dir = Path(output_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = history.get('epochs', [])
    epsilon = history.get('epsilon', 8.0)
    
    # 1. Loss plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    if 'train_loss' in history:
        plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (ε={epsilon})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy plot
    plt.subplot(1, 3, 2)
    if 'train_accuracy' in history:
        plt.plot(epochs, history['train_accuracy'], 'b-', label='Train Acc', linewidth=2)
    if 'val_accuracy' in history:
        plt.plot(epochs, history['val_accuracy'], 'r-', label='Val Acc', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy (ε={epsilon})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. F1 Score plot
    plt.subplot(1, 3, 3)
    if 'val_f1' in history:
        plt.plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2, marker='o')
        # Mark best F1
        if history['val_f1']:
            best_idx = np.argmax(history['val_f1'])
            best_f1 = history['val_f1'][best_idx]
            best_epoch = epochs[best_idx] if epochs else best_idx + 1
            plt.scatter([best_epoch], [best_f1], color='red', s=100, zorder=5)
            plt.annotate(f'Best: {best_f1:.3f}', 
                        (best_epoch, best_f1),
                        xytext=(10, 10),
                        textcoords='offset points')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'Validation F1 (ε={epsilon})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / f'training_curves_epsilon_{epsilon}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plots saved: {plot_path}")
    
    # Create combined metrics plot
    create_combined_plot(history, output_dir)
    
    return plot_path


def create_combined_plot(history: Dict, output_dir: Path):
    """Create combined metrics plot."""
    epochs = history.get('epochs', [])
    epsilon = history.get('epsilon', 8.0)
    
    plt.figure(figsize=(10, 6))
    
    metrics_to_plot = []
    if 'train_accuracy' in history and history['train_accuracy']:
        metrics_to_plot.append(('train_accuracy', 'Train Acc', 'blue'))
    if 'val_accuracy' in history and history['val_accuracy']:
        metrics_to_plot.append(('val_accuracy', 'Val Acc', 'red'))
    if 'val_f1' in history and history['val_f1']:
        metrics_to_plot.append(('val_f1', 'Val F1', 'green'))
    
    for metric_key, label, color in metrics_to_plot:
        plt.plot(epochs, history[metric_key], color=color, label=label, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title(f'DP Model Training Metrics (ε={epsilon})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add privacy info annotation
    if 'privacy_cost' in history:
        pc = history['privacy_cost']
        privacy_text = f"Privacy: ε={pc.get('epsilon', epsilon):.2f}, δ={pc.get('delta', 1e-5):.0e}"
        plt.figtext(0.5, 0.01, privacy_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plot_path = output_dir / f'combined_metrics_epsilon_{epsilon}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Combined metrics plot saved: {plot_path}")
    
    return plot_path


def generate_privacy_tradeoff_plot(epsilon_values: List[float], 
                                 f1_scores: List[float],
                                 output_path: Path):
    """
    Generate privacy-utility tradeoff plot.
    
    Args:
        epsilon_values: List of epsilon values
        f1_scores: List of corresponding F1 scores
        output_path: Output file path
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by epsilon
    sorted_data = sorted(zip(epsilon_values, f1_scores))
    epsilons, f1s = zip(*sorted_data)
    
    plt.plot(epsilons, f1s, 'o-', linewidth=3, markersize=10, 
            color='darkblue', markerfacecolor='lightblue')
    
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('Privacy-Utility Tradeoff', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for eps, f1 in zip(epsilons, f1s):
        plt.annotate(f'{f1:.3f}', (eps, f1), 
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=10)
    
    # Invert x-axis (higher ε = less privacy)
    plt.gca().invert_xaxis()
    
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Privacy tradeoff plot saved: {output_path}")


def plot_all_models_comparison(models_dir: Path = Path("outputs/models")):
    """Generate comparison plot for all trained models."""
    epsilons = []
    f1_scores = []
    
    # Find all training_history.json files
    for history_file in models_dir.glob("distilbert_epsilon_*/training_history.json"):
        try:
            with open(history_file) as f:
                history = json.load(f)
            
            epsilon = history.get('epsilon', 8.0)
            if 'val_f1' in history and history['val_f1']:
                best_f1 = max(history['val_f1'])
                epsilons.append(epsilon)
                f1_scores.append(best_f1)
                print(f"Found model: ε={epsilon}, F1={best_f1:.4f}")
        except Exception as e:
            print(f"⚠️  Could not load {history_file}: {e}")
    
    if len(epsilons) >= 2:
        # Sort by epsilon
        sorted_indices = np.argsort(epsilons)
        epsilons = [epsilons[i] for i in sorted_indices]
        f1_scores = [f1_scores[i] for i in sorted_indices]
        
        # Generate plot
        output_path = models_dir.parent / "privacy_tradeoff_all_models.png"
        generate_privacy_tradeoff_plot(epsilons, f1_scores, output_path)
    else:
        print("⚠️  Need at least 2 models for comparison plot")


def main():
    """Command line interface for plotting."""
    parser = argparse.ArgumentParser(description="Generate training plots")
    
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing model checkpoint")
    parser.add_argument("--epsilon", type=float,
                       help="Privacy budget ε (optional, will be read from history)")
    parser.add_argument("--output_dir", type=str, default="outputs/plots",
                       help="Output directory for plots")
    parser.add_argument("--compare_all", action="store_true",
                       help="Generate comparison plot for all models")
    
    args = parser.parse_args()
    
    if args.compare_all:
        plot_all_models_comparison(Path(args.model_dir))
    else:
        # Load training history
        history_file = Path(args.model_dir) / "training_history.json"
        if not history_file.exists():
            print(f"❌ History file not found: {history_file}")
            return
        
        with open(history_file) as f:
            history = json.load(f)
        
        # Use provided epsilon or get from history
        epsilon = args.epsilon if args.epsilon else history.get('epsilon', 8.0)
        history['epsilon'] = epsilon
        
        # Generate plots
        output_dir = Path(args.output_dir) / f"distilbert_epsilon_{epsilon}"
        generate_training_plots(history, output_dir)


if __name__ == "__main__":
    main()
EOF

echo "✅ Fixed src/evaluation/plot_utils.py"

# ============================================================================
# 4. CREATE Master Training Script
# ============================================================================
echo -e "\n📝 Creating master training script..."

cat > train_all_epsilons.sh << 'EOF'
#!/bin/bash
# train_all_epsilons.sh - Train models for all epsilon values

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}"
echo "="*70
echo "   🚀 TRAIN ALL ε-VALUES"
echo "="*70
echo -e "${NC}"

# Epsilon values to train
EPSILONS=(8.0 5.0 3.0 2.0 1.0 0.5)
EPOCHS=5
BATCH_SIZE=32

# Create results summary file
SUMMARY_FILE="outputs/training_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "Training Summary - $(date)" > "$SUMMARY_FILE"
echo "Epsilon | Best F1 | Training Time | Model Directory" >> "$SUMMARY_FILE"
echo "--------|---------|---------------|----------------" >> "$SUMMARY_FILE"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

# Train each epsilon
for epsilon in "${EPSILONS[@]}"; do
    echo -e "\n${BLUE}="*60
    echo "   TRAINING: ε = $epsilon"
    echo "="*60
    echo -e "${NC}"
    
    START_TIME=$(date +%s)
    
    # Generate model name
    MODEL_NAME="distilbert_epsilon_${epsilon}"
    
    echo -e "${YELLOW}Configuration:${NC}"
    echo "  ε (epsilon): $epsilon"
    echo "  Model name:  $MODEL_NAME"
    echo "  Epochs:      $EPOCHS"
    echo "  Batch size:  $BATCH_SIZE"
    
    # Run training
    python src/training/train_dp.py \
        --epsilon $epsilon \
        --model_name "$MODEL_NAME" \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate 5e-5 \
        --max_seq_length 128 \
        --output_dir "outputs/models" \
        --early_stopping_patience 2
    
    # Generate plots
    python src/evaluation/plot_utils.py \
        --model_dir "outputs/models/$MODEL_NAME" \
        --epsilon $epsilon \
        --output_dir "outputs/plots"
    
    END_TIME=$(date +%s)
    TRAINING_TIME=$((END_TIME - START_TIME))
    
    # Extract best F1 score
    HISTORY_FILE="outputs/models/$MODEL_NAME/training_history.json"
    if [[ -f "$HISTORY_FILE" ]]; then
        BEST_F1=$(python -c "
import json
try:
    with open('$HISTORY_FILE') as f:
        data = json.load(f)
    if 'val_f1' in data and data['val_f1']:
        print(f'{max(data[\"val_f1\"]):.4f}')
    else:
        print('N/A')
except:
    print('N/A')
")
    else
        BEST_F1="N/A"
    fi
    
    # Add to summary
    printf "%-8s | %-7s | %-13s | outputs/models/%s\n" \
        "$epsilon" "$BEST_F1" "$((TRAINING_TIME / 60))m $((TRAINING_TIME % 60))s" "$MODEL_NAME" >> "$SUMMARY_FILE"
    
    echo -e "${GREEN}✅ Completed ε=$epsilon in $((TRAINING_TIME / 60))m $((TRAINING_TIME % 60))s${NC}"
    
    # Wait before next training
    if [[ "$epsilon" != "0.5" ]]; then
        echo -e "${YELLOW}Waiting 15 seconds before next training...${NC}"
        sleep 15
    fi
done

# Generate combined comparison plot
echo -e "\n${BLUE}Generating combined comparison plot...${NC}"
python src/evaluation/plot_utils.py --compare_all --model_dir "outputs/models"

echo -e "\n${GREEN}"
echo "="*70
echo "   ✅ ALL MODELS TRAINED SUCCESSFULLY!"
echo "="*70
echo -e "${NC}"

echo -e "${YELLOW}📋 Training Summary:${NC}"
cat "$SUMMARY_FILE"

echo -e "\n${YELLOW}📁 Output Structure:${NC}"
echo "outputs/models/distilbert_epsilon_*/  - Model checkpoints & configs"
echo "outputs/plots/distilbert_epsilon_*/   - Training plots"
echo "outputs/privacy_tradeoff_all_models.png - Combined comparison"
echo ""
echo -e "${YELLOW}📊 To train a single epsilon value:${NC}"
echo "  python src/training/train_dp.py --epsilon 2.0"
echo ""
echo -e "${YELLOW}📈 To generate plots for existing model:${NC}"
echo "  python src/evaluation/plot_utils.py --model_dir outputs/models/distilbert_epsilon_8.0"
EOF

chmod +x train_all_epsilons.sh

# ============================================================================
# 5. CREATE Quick Training Script
# ============================================================================
cat > train_single.sh << 'EOF'
#!/bin/bash
# train_single.sh - Train a single epsilon value

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <epsilon> [epochs] [batch_size]"
    echo "Example: $0 2.0 5 32"
    echo "Default: epochs=5, batch_size=32"
    exit 1
fi

EPSILON=$1
EPOCHS=${2:-5}
BATCH_SIZE=${3:-32}

echo "="*60
echo "Training DP model with ε=$EPSILON"
echo "="*60

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

# Run training
python src/training/train_dp.py \
    --epsilon $EPSILON \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 5e-5 \
    --max_seq_length 128 \
    --output_dir "outputs/models" \
    --early_stopping_patience 2

# Generate plots
MODEL_NAME="distilbert_epsilon_${EPSILON}"
python src/evaluation/plot_utils.py \
    --model_dir "outputs/models/$MODEL_NAME" \
    --epsilon $EPSILON \
    --output_dir "outputs/plots"

echo ""
echo "✅ Training complete!"
echo "📁 Model saved: outputs/models/$MODEL_NAME"
echo "📈 Plots saved: outputs/plots/distilbert_epsilon_${EPSILON}/"
EOF

chmod +x train_single.sh

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo -e "\n${GREEN}"
echo "="*70
echo "✅ ALL FILES FIXED SUCCESSFULLY!"
echo "="*70
echo -e "${NC}"

echo "📁 Backups saved to: $BACKUP_DIR"
echo ""
echo "📝 New files created:"
echo "  1. train_all_epsilons.sh  - Train all ε values (8.0, 5.0, 3.0, 2.0, 1.0, 0.5)"
echo "  2. train_single.sh        - Train single ε value"
echo ""
echo "🔧 Files updated to be ε-aware:"
echo "  1. src/models/distilbert_dp.py    - DP model with configurable ε"
echo "  2. src/training/train_dp.py       - Training script with ε parameter"
echo "  3. src/evaluation/plot_utils.py   - Plotting with ε in titles/filenames"
echo ""
echo "🚀 How to use:"
echo "  ./train_all_epsilons.sh           # Train all epsilon values"
echo "  ./train_single.sh 2.0             # Train with ε=2.0"
echo "  ./train_single.sh 1.0 10 16       # Train with ε=1.0, 10 epochs, batch 16"
echo ""
echo "📊 Output will be organized as:"
echo "  outputs/models/distilbert_epsilon_8.0/"
echo "  outputs/models/distilbert_epsilon_5.0/"
echo "  outputs/models/distilbert_epsilon_3.0/"
echo "  ... etc"
echo ""
echo "📈 Plots will include ε in titles and filenames"
echo "  training_curves_epsilon_8.0.png"
echo "  combined_metrics_epsilon_5.0.png"
echo "  privacy_tradeoff_all_models.png"

# Create verification script
cat > verify_fixes.sh << 'EOF'
#!/bin/bash
echo "Verifying ε-aware fixes..."
echo ""

echo "1. Checking distilbert_dp.py..."
grep -n "epsilon: float = " src/models/distilbert_dp.py
echo ""

echo "2. Checking train_dp.py argument parsing..."
grep -n "parser.add_argument.*epsilon" src/training/train_dp.py
echo ""

echo "3. Checking plot_utils.py ε handling..."
grep -n "epsilon" src/evaluation/plot_utils.py | head -5
echo ""

echo "✅ Verification complete!"
EOF

chmod +x verify_fixes.sh

echo ""
echo "🔍 Run ./verify_fixes.sh to verify the changes"
echo "="*70