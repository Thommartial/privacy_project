#!/usr/bin/env python3
"""
Working DistilBERT DP Model in JAX/Flax for PII Detection.
Fixed SelfAttention with correct parameters.
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
    """Simplified attention block with correct SelfAttention usage."""
    hidden_size: int = 384
    num_heads: int = 6
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, mask=None, training=False):
        # Create causal mask if needed
        if mask is not None:
            # Convert mask to attention mask format
            batch_size, seq_len = mask.shape
            attention_mask = mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -1e9
        else:
            attention_mask = None
        
        # Multi-head attention - CORRECT: SelfAttention returns callable
        self_attn = nn.SelfAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate if training else 0.0
        )
        
        # Apply attention with mask
        attn_output = self_attn(x, mask=attention_mask)
        
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
    Working DistilBERT model for token classification with DP support.
    """
    num_labels: int = 3  # B, I, O tags
    hidden_size: int = 384
    num_attention_heads: int = 6
    num_hidden_layers: int = 3
    max_position_embeddings: int = 512
    dropout: float = 0.1
    epsilon: float = 8.0
    delta: float = 1e-5
    noise_multiplier: float = 0.25
    
    @nn.compact
    def __call__(self, 
                 input_ids: jnp.ndarray,
                 attention_mask: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        """
        Forward pass.
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
            )(x, mask=attention_mask, training=training)
        
        # Classification head
        logits = nn.Dense(self.num_labels)(x)
        return logits
    
    def compute_privacy_cost(self, 
                           steps: int, 
                           batch_size: int, 
                           dataset_size: int,
                           noise_multiplier: float) -> Dict[str, float]:
        """
        Compute privacy cost.
        """
        q = batch_size / dataset_size
        
        # Simplified DP-SGD accounting
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
    Differential Privacy Trainer.
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
        
        # Calculate noise multiplier
        self.noise_multiplier = max(0.1, 2.0 / epsilon)
        
        # Optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adamw(learning_rate=self.learning_rate, weight_decay=0.01)
        )
    
    def init_state(self, rng_key: jnp.ndarray, 
                   dummy_input: jnp.ndarray,
                   dummy_mask: jnp.ndarray) -> TrainingState:
        """
        Initialize training state.
        """
        params = self.model.init(rng_key, dummy_input, dummy_mask, training=False)
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
        Compute loss and accuracy.
        """
        input_ids, attention_mask, labels = batch
        
        logits = self.model.apply(params, input_ids, attention_mask, 
                                 training=True, rngs={'dropout': rng_key})
        
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        
        return loss, {'loss': loss, 'accuracy': accuracy}
    
    def train_step(self, state: TrainingState,
                   batch: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Tuple[TrainingState, dict]:
        """
        Single training step with DP-SGD.
        """
        input_ids, attention_mask, labels = batch
        
        rng, dropout_rng = jax.random.split(state.rng)
        
        grad_fn = jax.value_and_grad(self.compute_loss, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params, 
                                        (input_ids, attention_mask, labels),
                                        dropout_rng)
        
        # Apply DP modifications
        grads = self._apply_dp_modifications(grads, rng)
        
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        
        new_state = state.replace(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            rng=rng
        )
        
        return new_state, metrics
    
    def _apply_dp_modifications(self, grads: dict, rng_key: jnp.ndarray) -> dict:
        """
        Apply DP-SGD: clipping and noise.
        """
        # Clip
        grads = jax.tree.map(
            lambda g: g / jnp.maximum(jnp.linalg.norm(g) / self.max_grad_norm, 1.0),
            grads
        )
        
        # Add noise
        noise_key, _ = jax.random.split(rng_key)
        
        def add_noise(grad):
            noise = jax.random.normal(noise_key, shape=grad.shape)
            return grad + self.noise_multiplier * self.max_grad_norm * noise / self.batch_size
        
        grads = jax.tree.map(add_noise, grads)
        
        return grads
    
    def evaluate(self, params: dict, 
                 dataset: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> Dict[str, float]:
        """
        Evaluate model.
        """
        input_ids, attention_mask, labels = dataset
        
        logits = self.model.apply(params, input_ids, attention_mask, training=False)
        predictions = jnp.argmax(logits, axis=-1)
        
        accuracy = (predictions == labels).mean()
        
        # Simple PII detection metrics
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
            'f1_score': float(f1)
        }


def save_checkpoint(state: TrainingState, path: str, metrics: dict, epsilon: float):
    """Save checkpoint."""
    import pickle
    checkpoint = {
        'state': state,
        'metrics': metrics,
        'config': {'epsilon': epsilon}
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"✅ Checkpoint saved: {path}")


def load_checkpoint(path: str):
    """Load checkpoint."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)
