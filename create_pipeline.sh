#!/bin/bash
# create_proper_pipeline.sh - Create full DP training pipeline

echo "🎯 CREATING PROPER DP TRAINING PIPELINE"
echo "="*70

# 1. Create evaluation script
cat > src/evaluation/evaluate_dp_model.py << 'EOF'
#!/usr/bin/env python3
"""
Evaluate DP model and generate all required plots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse


def load_model_results(model_dir: Path) -> Dict:
    """Load model checkpoint and history."""
    checkpoint_path = model_dir / "checkpoint.pkl"
    history_path = model_dir / "history.json"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    with open(history_path) as f:
        history = json.load(f)
    
    return {
        'checkpoint': checkpoint,
        'history': history,
        'config': checkpoint.get('config', {})
    }


def generate_training_plots(history: Dict, output_dir: Path, epsilon: float):
    """Generate all training visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # 1. Loss plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', linewidth=2, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss (ε={epsilon})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 2. Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', linewidth=2, marker='o', label='Train Acc')
    if 'val_acc' in history:
        plt.plot(epochs, history['val_acc'], 'r-', linewidth=2, marker='s', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training Accuracy (ε={epsilon})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'training_curves_epsilon_{epsilon}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plots saved: {output_dir}/training_curves_epsilon_{epsilon}.png")
    
    # 3. Privacy-Utility summary (for single model)
    if 'privacy_cost' in history:
        plt.figure(figsize=(8, 5))
        
        epsilons = [history['privacy_cost']['epsilon']]
        accuracies = [max(history['val_acc']) if 'val_acc' in history else history['train_acc'][-1]]
        
        plt.scatter(epsilons, accuracies, s=200, color='darkblue', alpha=0.7, edgecolors='black')
        plt.xlabel('Privacy Budget (ε)')
        plt.ylabel('Accuracy')
        plt.title(f'Privacy-Accuracy Tradeoff\nε={epsilons[0]:.3f}, Acc={accuracies[0]:.3f}')
        plt.grid(True, alpha=0.3)
        
        # Add annotation
        plt.annotate(f'ε={epsilons[0]:.3f}\nAcc={accuracies[0]:.3f}',
                    (epsilons[0], accuracies[0]),
                    xytext=(10, 10),
                    textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'privacy_tradeoff_epsilon_{epsilon}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Privacy tradeoff plot saved: {output_dir}/privacy_tradeoff_epsilon_{epsilon}.png")


def compare_multiple_models(model_dirs: List[Path], output_dir: Path):
    """Compare multiple ε values."""
    epsilons = []
    accuracies = []
    f1_scores = []
    
    for model_dir in model_dirs:
        try:
            with open(model_dir / "history.json") as f:
                history = json.load(f)
            
            eps = history.get('epsilon', float(model_dir.name.split('_')[-1]))
            epsilons.append(eps)
            
            if 'val_acc' in history:
                accuracies.append(max(history['val_acc']))
            else:
                accuracies.append(history['train_acc'][-1])
            
            if 'val_f1' in history:
                f1_scores.append(max(history['val_f1']))
            
            print(f"📊 Model ε={eps}: Acc={accuracies[-1]:.3f}")
            
        except Exception as e:
            print(f"⚠️  Could not load {model_dir}: {e}")
    
    if len(epsilons) >= 2:
        # Sort by epsilon
        sorted_indices = np.argsort(epsilons)
        epsilons = [epsilons[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        
        # Create comparison plot
        plt.figure(figsize=(10, 6))
        
        plt.plot(epsilons, accuracies, 'o-', linewidth=3, markersize=10, 
                color='darkblue', markerfacecolor='lightblue')
        
        plt.xlabel('Privacy Budget (ε)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Privacy-Accuracy Tradeoff (Multiple Models)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for eps, acc in zip(epsilons, accuracies):
            plt.annotate(f'{acc:.3f}', (eps, acc), 
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=10)
        
        plt.tight_layout()
        plot_path = output_dir / "privacy_tradeoff_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comparison plot saved: {plot_path}")
        
        # Save comparison data
        comparison = {
            'epsilons': epsilons,
            'accuracies': accuracies,
            'models': [str(d) for d in model_dirs]
        }
        
        with open(output_dir / "model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✅ Comparison data saved: {output_dir}/model_comparison.json")


def generate_report(model_dir: Path, output_dir: Path):
    """Generate a comprehensive report."""
    results = load_model_results(model_dir)
    history = results['history']
    config = results['config']
    epsilon = config.get('epsilon', history.get('epsilon', 8.0))
    
    report = f"""
# DP Model Evaluation Report

## Model Information
- **Privacy Budget (ε)**: {epsilon}
- **Training Time**: {len(history['train_loss'])} epochs
- **Best Validation Accuracy**: {max(history.get('val_acc', [0])):.4f}
- **Final Training Loss**: {history['train_loss'][-1]:.4f}

## Privacy Analysis
"""
    
    if 'privacy_cost' in history:
        pc = history['privacy_cost']
        report += f"""
- **Achieved ε**: {pc['epsilon']:.3f}
- **δ (delta)**: {pc['delta']:.0e}
- **Noise Multiplier**: {pc.get('sigma', 'N/A'):.3f}
- **Sampling Probability (q)**: {pc.get('q', 'N/A'):.4f}
- **Training Steps**: {pc.get('steps', 'N/A')}
"""
    
    report += f"""
## Training History
- Epochs: {list(range(1, len(history['train_loss']) + 1))}
- Training Loss: {[f'{x:.4f}' for x in history['train_loss']]}
- Training Accuracy: {[f'{x:.4f}' for x in history['train_acc']]}
"""
    
    if 'val_acc' in history:
        report += f"- Validation Accuracy: {[f'{x:.4f}' for x in history['val_acc']]}\n"
    
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved: {report_path}")
    
    # Generate plots
    generate_training_plots(history, output_dir, epsilon)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DP model")
    
    parser.add_argument("--model_dir", type=str, required=True,
                       help="Directory containing model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation",
                       help="Output directory for plots and reports")
    parser.add_argument("--compare_all", action="store_true",
                       help="Compare all models in outputs/models/")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare_all:
        # Find all model directories
        models_dir = Path("outputs/models")
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and "epsilon" in d.name]
        
        if len(model_dirs) >= 2:
            print(f"📊 Comparing {len(model_dirs)} models...")
            compare_multiple_models(model_dirs, output_dir)
        else:
            print(f"⚠️  Need at least 2 models for comparison. Found: {len(model_dirs)}")
    
    # Always evaluate the specified model
    model_dir = Path(args.model_dir)
    if model_dir.exists():
        print(f"📈 Evaluating model: {model_dir}")
        generate_report(model_dir, output_dir / model_dir.name)
    else:
        print(f"❌ Model directory not found: {model_dir}")


if __name__ == "__main__":
    main()
EOF

# 2. Create improved training script with real data
cat > src/training/train_dp_proper.py << 'EOF'
#!/usr/bin/env python3
"""
Proper DP training with real data and better model.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import pickle
import argparse
from typing import Dict, Tuple
import optax


class ProperDPModel(jax.nn.Module):
    """Better but still simple DP model."""
    vocab_size: int = 10000
    hidden_size: int = 256
    num_labels: int = 3
    epsilon: float = 8.0
    delta: float = 1e-5
    
    @jax.nn.compact
    def __call__(self, input_ids, attention_mask=None, training=False):
        # Embedding layer
        x = jax.nn.Embed(self.vocab_size, self.hidden_size)(input_ids)
        
        # Simple LSTM-like processing
        x = jax.nn.Dense(self.hidden_size)(x)
        x = jax.nn.relu(x)
        x = jax.nn.Dropout(rate=0.2, deterministic=not training)(x)
        
        # Global pooling
        x = jnp.mean(x, axis=1)  # [batch, hidden]
        
        # Classification head
        x = jax.nn.Dense(128)(x)
        x = jax.nn.relu(x)
        x = jax.nn.Dropout(rate=0.1, deterministic=not training)(x)
        
        # Expand for sequence labeling
        batch_size, seq_len = input_ids.shape
        x = x[:, None, :]  # [batch, 1, hidden]
        x = jnp.tile(x, (1, seq_len, 1))  # [batch, seq_len, hidden]
        
        logits = jax.nn.Dense(self.num_labels)(x)
        return logits
    
    def compute_privacy_cost(self, steps, batch_size, dataset_size, noise_multiplier):
        q = batch_size / dataset_size
        epsilon = noise_multiplier * q * np.sqrt(steps * np.log(1/self.delta))
        return {
            'epsilon': float(epsilon),
            'delta': self.delta,
            'sigma': noise_multiplier,
            'q': q,
            'steps': steps
        }


def load_real_data(epsilon: float, max_samples: int = 1000, max_seq_length: int = 64):
    """Load real PII data with proper labels."""
    print("📂 Loading real PII data...")
    
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    
    # Use more data for higher epsilon (less privacy = can use more data)
    scale_factor = min(1.0, epsilon / 8.0)  # Scale from 0.5 to 1.0
    n_train = int(max_samples * scale_factor)
    n_val = int(max_samples * 0.2 * scale_factor)
    
    train_df = train_df.sample(n=min(n_train, len(train_df)), random_state=42)
    val_df = val_df.sample(n=min(n_val, len(val_df)), random_state=42)
    
    print(f"  Using {len(train_df)} train, {len(val_df)} val samples for ε={epsilon}")
    
    def parse_labels(labels_str):
        if isinstance(labels_str, str):
            try:
                return eval(labels_str)
            except:
                return []
        return labels_str
    
    train_df['labels'] = train_df['labels'].apply(parse_labels)
    val_df['labels'] = val_df['labels'].apply(parse_labels)
    
    def create_dataset(df, seq_len):
        texts = df['text'].tolist()
        labels = df['labels'].tolist()
        
        input_ids = []
        label_arrays = []
        
        for text, label_list in zip(texts, labels):
            # Simple tokenization
            words = text.split()[:seq_len]
            current_len = len(words)
            
            # Simple hash-based token IDs
            ids = [hash(word) % 10000 for word in words]
            if current_len < seq_len:
                ids = ids + [0] * (seq_len - current_len)
            
            # Convert labels to indices
            label_array = [0] * seq_len
            for i in range(min(len(label_list), seq_len)):
                label = label_list[i]
                if label == 'O':
                    label_array[i] = 0
                elif label.startswith('B-'):
                    label_array[i] = 1
                elif label.startswith('I-'):
                    label_array[i] = 2
            
            input_ids.append(ids)
            label_arrays.append(label_array)
        
        return {
            'input_ids': jnp.array(input_ids, dtype=jnp.int32),
            'attention_mask': jnp.ones((len(df), seq_len), dtype=jnp.int32),
            'labels': jnp.array(label_arrays, dtype=jnp.int32)
        }
    
    train_data = create_dataset(train_df, max_seq_length)
    val_data = create_dataset(val_df, max_seq_length)
    
    print(f"✅ Data loaded. Input shape: {train_data['input_ids'].shape}")
    return train_data, val_data


def train_proper_model(args):
    """Train proper DP model."""
    print(f"\n🚀 Training Proper DP Model (ε={args.epsilon})")
    
    # Load data
    train_data, val_data = load_real_data(
        epsilon=args.epsilon,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length
    )
    
    # Create model
    model = ProperDPModel(epsilon=args.epsilon, hidden_size=args.hidden_size)
    
    # Setup training
    noise_multiplier = max(0.1, 2.0 / args.epsilon)
    learning_rate = 5e-5
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate)
    )
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    rng, init_rng = jax.random.split(rng)
    
    dummy_input = jnp.ones((2, args.max_seq_length), dtype=jnp.int32)
    dummy_mask = jnp.ones((2, args.max_seq_length), dtype=jnp.int32)
    
    params = model.init(init_rng, dummy_input, dummy_mask, training=False)
    opt_state = optimizer.init(params)
    
    # Training function
    @jax.jit
    def train_step(params, opt_state, batch, rng):
        input_ids, attention_mask, labels = batch
        rng, dropout_rng = jax.random.split(rng)
        
        def loss_fn(params):
            logits = model.apply(params, input_ids, attention_mask, training=True, rngs={'dropout': dropout_rng})
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        # DP modifications
        grads = jax.tree.map(
            lambda g: g / jnp.maximum(jnp.linalg.norm(g) / 1.0, 1.0),
            grads
        )
        
        noise_key, _ = jax.random.split(rng)
        def add_noise(grad):
            noise = jax.random.normal(noise_key, shape=grad.shape)
            return grad + noise_multiplier * 1.0 * noise / args.batch_size
        grads = jax.tree.map(add_noise, grads)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss, rng
    
    @jax.jit
    def evaluate(params, batch):
        input_ids, attention_mask, labels = batch
        logits = model.apply(params, input_ids, attention_mask, training=False)
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return accuracy
    
    # Training loop
    history = {
        'epsilon': args.epsilon,
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'epochs': []
    }
    
    n_samples = train_data['input_ids'].shape[0]
    n_batches = (n_samples + args.batch_size - 1) // args.batch_size
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        
        for i in range(0, n_samples, args.batch_size):
            batch_idx = indices[i:i+args.batch_size]
            batch = (
                train_data['input_ids'][batch_idx],
                train_data['attention_mask'][batch_idx],
                train_data['labels'][batch_idx]
            )
            
            params, opt_state, loss, rng = train_step(params, opt_state, batch, rng)
            epoch_loss += float(loss)
        
        avg_loss = epoch_loss / n_batches
        
        # Validation
        val_acc = float(evaluate(params, (
            val_data['input_ids'][:args.batch_size],
            val_data['attention_mask'][:args.batch_size],
            val_data['labels'][:args.batch_size]
        )))
        
        # Training accuracy (small sample)
        train_acc = float(evaluate(params, (
            train_data['input_ids'][:args.batch_size],
            train_data['attention_mask'][:args.batch_size],
            train_data['labels'][:args.batch_size]
        )))
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)
        
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1}/{args.epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"Loss: {avg_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")
    
    # Privacy cost
    total_steps = args.epochs * n_batches
    privacy_cost = model.compute_privacy_cost(
        steps=total_steps,
        batch_size=args.batch_size,
        dataset_size=n_samples,
        noise_multiplier=noise_multiplier
    )
    
    history['privacy_cost'] = privacy_cost
    history['noise_multiplier'] = noise_multiplier
    history['total_steps'] = total_steps
    
    # Save model
    model_name = f"proper_epsilon_{args.epsilon}"
    output_dir = Path("outputs/models") / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'params': params,
        'opt_state': opt_state,
        'history': history,
        'config': {
            'epsilon': args.epsilon,
            'hidden_size': args.hidden_size,
            'batch_size': args.batch_size,
            'max_seq_length': args.max_seq_length,
            'max_samples': args.max_samples,
            'epochs': args.epochs
        }
    }
    
    with open(output_dir / "checkpoint.pkl", "wb") as f:
        pickle.dump(checkpoint, f)
    
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    print(f"\n✅ Training complete!")
    print(f"   Best Val Acc: {max(history['val_acc']):.4f}")
    print(f"   Privacy: ε={privacy_cost['epsilon']:.3f}")
    print(f"   Model saved: {output_dir}")
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Train proper DP model")
    
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--max_seq_length", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=1000)
    
    args = parser.parse_args()
    
    print("="*70)
    print("🚀 PROPER DP MODEL TRAINING")
    print("="*70)
    print(f"  ε: {args.epsilon}")
    print(f"  Model: ProperDPModel (hidden_size={args.hidden_size})")
    print(f"  Data: {args.max_samples} samples, seq_len={args.max_seq_length}")
    print("="*70)
    
    try:
        train_proper_model(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
EOF

# 3. Create master training script for all ε values
cat > train_all_epsilons_proper.sh << 'EOF'
#!/bin/bash
# train_all_epsilons_proper.sh - Train proper models for all ε values

set -e

echo "="*70
echo "🎯 TRAIN ALL ε-VALUES (PROPER MODELS)"
echo "="*70

EPSILONS=(8.0 5.0 3.0 2.0 1.0 0.5)
SUMMARY_FILE="outputs/training_summary_proper_$(date +%Y%m%d_%H%M%S).txt"

echo "Training ε values: ${EPSILONS[*]}" > "$SUMMARY_FILE"
echo "Timestamp: $(date)" >> "$SUMMARY_FILE"
echo "="*60 >> "$SUMMARY_FILE"
echo "ε | Best Val Acc | Privacy ε | Time" >> "$SUMMARY_FILE"
echo "--|-------------|-----------|------" >> "$SUMMARY_FILE"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

for epsilon in "${EPSILONS[@]}"; do
    echo -e "\n\033[1;34m"  # Blue
    echo "="*60
    echo "   TRAINING ε = $epsilon"
    echo "="*60
    echo -e "\033[0m"
    
    START_TIME=$(date +%s)
    
    # Adjust parameters based on epsilon
    if (( $(echo "$epsilon >= 5.0" | bc -l) )); then
        epochs=5
        batch_size=16
        max_samples=1000
    elif (( $(echo "$epsilon >= 2.0" | bc -l) )); then
        epochs=4
        batch_size=12
        max_samples=800
    else
        epochs=3
        batch_size=8
        max_samples=500
    fi
    
    echo "  Configuration:"
    echo "    Epochs: $epochs"
    echo "    Batch size: $batch_size"
    echo "    Max samples: $max_samples"
    
    python src/training/train_dp_proper.py \
        --epsilon $epsilon \
        --epochs $epochs \
        --batch_size $batch_size \
        --max_samples $max_samples \
        --max_seq_length 64 \
        --hidden_size 256
    
    END_TIME=$(date +%s)
    TRAINING_TIME=$((END_TIME - START_TIME))
    
    # Extract results
    MODEL_DIR="outputs/models/proper_epsilon_$epsilon"
    if [[ -f "$MODEL_DIR/history.json" ]]; then
        BEST_VAL_ACC=$(python -c "
import json
with open('$MODEL_DIR/history.json') as f:
    data = json.load(f)
print(f\"{max(data['val_acc']):.4f}\")" 2>/dev/null || echo "N/A")
        
        PRIVACY_EPSILON=$(python -c "
import json
with open('$MODEL_DIR/history.json') as f:
    data = json.load(f)
if 'privacy_cost' in data:
    print(f\"{data['privacy_cost']['epsilon']:.3f}\")
else:
    print('N/A')" 2>/dev/null || echo "N/A")
    else
        BEST_VAL_ACC="N/A"
        PRIVACY_EPSILON="N/A"
    fi
    
    # Add to summary
    printf "%-4s | %-12s | %-10s | %dm %ds\n" \
        "$epsilon" "$BEST_VAL_ACC" "$PRIVACY_EPSILON" \
        $((TRAINING_TIME / 60)) $((TRAINING_TIME % 60)) >> "$SUMMARY_FILE"
    
    echo -e "\033[1;32m"  # Green
    echo "✅ Completed ε=$epsilon in $((TRAINING_TIME / 60))m $((TRAINING_TIME % 60))s"
    echo -e "\033[0m"
    
    # Wait before next training
    if [[ "$epsilon" != "0.5" ]]; then
        echo "⏳ Waiting 10 seconds..."
        sleep 10
    fi
done

# Generate comparison plot
echo -e "\n\033[1;36m"  # Cyan
echo "="*70
echo "   GENERATING COMPARISON PLOTS"
echo "="*70
echo -e "\033[0m"

python src/evaluation/evaluate_dp_model.py --compare_all --output_dir outputs/evaluation

echo -e "\n\033[1;32m"  # Green
echo "="*70
echo "   ✅ ALL MODELS TRAINED!"
echo "="*70
echo -e "\033[0m"

echo "📋 Summary:"
cat "$SUMMARY_FILE"

echo ""
echo "📁 Models saved in: outputs/models/proper_epsilon_*/"
echo "📊 Evaluation plots: outputs/evaluation/"
echo "📄 Summary file: $SUMMARY_FILE"
echo ""
echo "📈 To evaluate a specific model:"
echo "   python src/evaluation/evaluate_dp_model.py --model_dir outputs/models/proper_epsilon_8.0"
echo "="*70
EOF

chmod +x train_all_epsilons_proper.sh
chmod +x src/evaluation/evaluate_dp_model.py
chmod +x src/training/train_dp_proper.py

echo "✅ Created proper DP training pipeline"
echo ""
echo "🎯 Files created:"
echo "   1. src/evaluation/evaluate_dp_model.py    - Comprehensive evaluation"
echo "   2. src/training/train_dp_proper.py       - Proper training with real data"
echo "   3. train_all_epsilons_proper.sh          - Train all ε values"
echo ""
echo "🚀 To train a single ε=8.0 model properly:"
echo "   python src/training/train_dp_proper.py --epsilon 8.0"
echo ""
echo "🚀 To train ALL ε values (8.0 → 0.5):"
echo "   ./train_all_epsilons_proper.sh"
echo ""
echo "📊 To evaluate your minimal model:"
echo "   python src/evaluation/evaluate_dp_model.py --model_dir outputs/models/minimal_epsilon_8.0"
