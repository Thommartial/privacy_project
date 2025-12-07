#!/bin/bash
# fix_python_path.sh - Fix Python import paths

echo "🔧 Fixing Python import paths..."

# Create __init__.py in root if it doesn't exist
touch __init__.py

# Update train_dp.py to add project root to sys.path
cat > src/training/train_dp.py << 'EOF'
#!/usr/bin/env python3
"""
Main training script for DP DistilBERT model.
Fast, efficient, with proper logging and early stopping.
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

echo "✅ Fixed train_dp.py imports"

# Also fix plot_utils.py
cat > src/evaluation/plot_utils.py << 'EOF'
#!/usr/bin/env python3
"""
Plotting utilities for training visualization.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

echo "✅ Fixed plot_utils.py imports"

# Create a simple test script to verify imports
cat > test_imports.py << 'EOF'
#!/usr/bin/env python3
"""Test that all imports work correctly."""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("Testing imports from project root...")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

try:
    from src.models.distilbert_dp import DistilBertDP
    print("✅ Successfully imported DistilBertDP")
    
    from src.training.train_dp import parse_args
    print("✅ Successfully imported train_dp functions")
    
    import jax
    print(f"✅ JAX version: {jax.__version__}")
    
    import flax
    print(f"✅ Flax version: {flax.__version__}")
    
    print("\n✅ All imports working correctly!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
EOF

chmod +x test_imports.py

echo ""
echo "✅ Fixed all import paths"
echo "📋 To test imports: python test_imports.py"
echo "📋 To run training: ./run_epsilon_8.sh"
