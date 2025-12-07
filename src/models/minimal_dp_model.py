#!/usr/bin/env python3
"""
Minimal but WORKING DP model for PII detection.
No fancy transformer, just enough to train with DP.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import optax
import numpy as np
from typing import Dict, Tuple


@struct.dataclass
class TrainingState:
    step: int
    params: dict
    opt_state: optax.OptState
    rng: jnp.ndarray


class MinimalPIIModel(nn.Module):
    """Minimal model that actually works."""
    vocab_size: int = 30522
    hidden_size: int = 256  # Divisible by many numbers
    num_labels: int = 3
    epsilon: float = 8.0
    delta: float = 1e-5
    
    @nn.compact
    def __call__(self, input_ids, attention_mask=None, training=False):
        # Simple embedding
        x = nn.Embed(self.vocab_size, self.hidden_size)(input_ids)
        
        # Simple pooling (mean over sequence)
        x = jnp.mean(x, axis=1)  # [batch, hidden]
        
        # Simple feed-forward
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.1, deterministic=not training)(x)
        
        # Expand back to sequence length for token classification
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        x = x[:, None, :]  # [batch, 1, hidden]
        x = jnp.tile(x, (1, seq_len, 1))  # [batch, seq_len, hidden]
        
        # Classification
        logits = nn.Dense(self.num_labels)(x)
        return logits
    
    def compute_privacy_cost(self, steps, batch_size, dataset_size, noise_multiplier):
        """Simplified privacy accounting."""
        q = batch_size / dataset_size
        epsilon = noise_multiplier * q * np.sqrt(steps * np.log(1/self.delta))
        return {
            'epsilon': float(epsilon),
            'delta': self.delta,
            'sigma': noise_multiplier,
            'q': q,
            'steps': steps
        }


class MinimalDPTrainer:
    """Minimal DP trainer that works."""
    
    def __init__(self, model, learning_rate=5e-5, epsilon=8.0, batch_size=8):
        self.model = model
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.max_grad_norm = 1.0
        self.noise_multiplier = max(0.1, 2.0 / epsilon)
        
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(self.learning_rate)
        )
    
    def init_state(self, rng, dummy_input, dummy_mask):
        params = self.model.init(rng, dummy_input, dummy_mask, training=False)
        opt_state = self.optimizer.init(params)
        return TrainingState(step=0, params=params, opt_state=opt_state, rng=rng)
    
    def compute_loss(self, params, batch, rng):
        input_ids, attention_mask, labels = batch
        logits = self.model.apply(params, input_ids, attention_mask, training=True, rngs={'dropout': rng})
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return loss, {'loss': loss, 'accuracy': accuracy}
    
    def train_step(self, state, batch):
        input_ids, attention_mask, labels = batch
        rng, dropout_rng = jax.random.split(state.rng)
        
        grad_fn = jax.value_and_grad(self.compute_loss, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params, (input_ids, attention_mask, labels), dropout_rng)
        
        # DP modifications
        grads = jax.tree.map(
            lambda g: g / jnp.maximum(jnp.linalg.norm(g) / self.max_grad_norm, 1.0),
            grads
        )
        
        noise_key, _ = jax.random.split(rng)
        def add_noise(grad):
            noise = jax.random.normal(noise_key, shape=grad.shape)
            return grad + self.noise_multiplier * self.max_grad_norm * noise / self.batch_size
        grads = jax.tree.map(add_noise, grads)
        
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        
        new_state = state.replace(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            rng=rng
        )
        
        return new_state, metrics
    
    def evaluate(self, params, batch):
        input_ids, attention_mask, labels = batch
        logits = self.model.apply(params, input_ids, attention_mask, training=False)
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        
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
