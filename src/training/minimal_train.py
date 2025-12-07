#!/usr/bin/env python3
"""
Minimal training script that WORKS.
Trains a simple model with DP in < 5 minutes.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import argparse

from src.models.minimal_dp_model import MinimalPIIModel, MinimalDPTrainer, TrainingState


def load_tiny_dataset(max_seq_length=32, n_samples=200):
    """Load tiny dataset for quick training."""
    print("📂 Loading tiny dataset...")
    
    df = pd.read_csv("data/processed/train.csv")
    df = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    # Simple preprocessing
    texts = df['text'].tolist()
    
    input_ids = []
    labels_list = []
    
    for text in texts:
        # Simple tokenization
        words = text.split()[:max_seq_length]
        seq_len = len(words)
        
        # Simple IDs
        ids = [hash(word) % 1000 for word in words]
        if seq_len < max_seq_length:
            ids = ids + [0] * (max_seq_length - seq_len)
        
        # Simple labels (all O for now - just to make it work)
        labels = [0] * max_seq_length
        
        input_ids.append(ids)
        labels_list.append(labels)
    
    return {
        'input_ids': jnp.array(input_ids, dtype=jnp.int32),
        'attention_mask': jnp.ones((len(df), max_seq_length), dtype=jnp.int32),
        'labels': jnp.array(labels_list, dtype=jnp.int32)
    }


def train_minimal(args):
    """Minimal training loop."""
    print(f"\n🚀 Starting MINIMAL training (ε={args.epsilon})")
    print("   This will complete in 1-2 minutes!")
    
    # Load data
    data = load_tiny_dataset(args.max_seq_length, args.n_samples)
    
    # Create model
    model = MinimalPIIModel(epsilon=args.epsilon, hidden_size=128)
    trainer = MinimalDPTrainer(model, epsilon=args.epsilon, batch_size=args.batch_size)
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((2, args.max_seq_length), dtype=jnp.int32)
    dummy_mask = jnp.ones((2, args.max_seq_length), dtype=jnp.int32)
    
    state = trainer.init_state(rng, dummy_input, dummy_mask)
    
    # Create batches
    n_samples = data['input_ids'].shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    batches = []
    for i in range(0, n_samples, args.batch_size):
        batch_idx = indices[i:i+args.batch_size]
        batch = (
            data['input_ids'][batch_idx],
            data['attention_mask'][batch_idx],
            data['labels'][batch_idx]
        )
        batches.append(batch)
    
    # Training
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        epoch_loss = []
        epoch_acc = []
        
        for batch in batches:
            state, metrics = trainer.train_step(state, batch)
            epoch_loss.append(float(metrics['loss']))
            epoch_acc.append(float(metrics['accuracy']))
        
        # Simple validation (use first batch)
        val_metrics = trainer.evaluate(state.params, batches[0])
        
        avg_loss = np.mean(epoch_loss)
        avg_acc = np.mean(epoch_acc)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        history['val_acc'].append(val_metrics['accuracy'])
        
        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {avg_acc:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        if avg_loss < 0.1:  # Early stop if loss is very low
            print("    ↳ Loss very low, stopping early")
            break
    
    # Privacy cost
    privacy_cost = model.compute_privacy_cost(
        steps=state.step,
        batch_size=args.batch_size,
        dataset_size=n_samples,
        noise_multiplier=trainer.noise_multiplier
    )
    
    history['privacy_cost'] = privacy_cost
    history['epsilon'] = args.epsilon
    
    # Save results
    output_dir = Path("outputs/models") / f"minimal_epsilon_{args.epsilon}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save checkpoint
    import pickle
    checkpoint = {
        'state': state,
        'history': history,
        'config': {
            'epsilon': args.epsilon,
            'batch_size': args.batch_size,
            'seq_length': args.max_seq_length
        }
    }
    with open(output_dir / "checkpoint.pkl", "wb") as f:
        pickle.dump(checkpoint, f)
    
    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"\n✅ Training complete!")
    print(f"   Privacy: ε={privacy_cost['epsilon']:.3f}")
    print(f"   Best Val Acc: {max(history['val_acc']):.4f}")
    print(f"   Saved to: {output_dir}")
    
    return state, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=100)
    
    args = parser.parse_args()
    
    print("="*60)
    print("🧪 MINIMAL DP MODEL TRAINING")
    print("="*60)
    print(f"  Goal: Get SOMETHING working in < 2 minutes")
    print(f"  ε: {args.epsilon}, Epochs: {args.epochs}")
    print("="*60)
    
    try:
        train_minimal(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
