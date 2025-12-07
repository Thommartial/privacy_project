#!/bin/bash
# fix_evaluation.sh - Fix directory creation in evaluation script

echo "🔧 Fixing evaluation script..."

# Fix the directory creation issue
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
    # Try different possible file locations
    checkpoint_path = model_dir / "checkpoint.pkl"
    history_path = model_dir / "history.json"
    
    # If not found, try other names
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "model_checkpoint.pkl"
    
    if not history_path.exists():
        history_path = model_dir / "training_history.json"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found in {model_dir}")
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
    except:
        checkpoint = {}
    
    try:
        with open(history_path) as f:
            history = json.load(f)
    except:
        history = {}
    
    return {
        'checkpoint': checkpoint,
        'history': history,
        'config': checkpoint.get('config', {})
    }


def generate_training_plots(history: Dict, output_dir: Path, epsilon: float):
    """Generate all training visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'train_loss' not in history or not history['train_loss']:
        print("⚠️  No training history found")
        return
    
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
    if 'train_acc' in history:
        plt.plot(epochs, history['train_acc'], 'b-', linewidth=2, marker='o', label='Train Acc')
    
    if 'val_acc' in history and history['val_acc']:
        plt.plot(epochs, history['val_acc'], 'r-', linewidth=2, marker='s', label='Val Acc')
    elif 'train_accuracy' in history:
        plt.plot(epochs, history['train_accuracy'], 'b-', linewidth=2, marker='o', label='Train Acc')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training Accuracy (ε={epsilon})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plot_path = output_dir / f'training_curves_epsilon_{epsilon}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training plots saved: {plot_path}")
    
    # 3. Privacy-Utility summary
    if 'privacy_cost' in history:
        plt.figure(figsize=(8, 5))
        
        eps = history['privacy_cost'].get('epsilon', epsilon)
        if 'val_acc' in history and history['val_acc']:
            acc = max(history['val_acc'])
        elif 'train_acc' in history and history['train_acc']:
            acc = max(history['train_acc'])
        else:
            acc = 0.5
        
        plt.scatter([eps], [acc], s=200, color='darkblue', alpha=0.7, edgecolors='black')
        plt.xlabel('Privacy Budget (ε)')
        plt.ylabel('Accuracy')
        plt.title(f'Privacy-Accuracy Tradeoff\nε={eps:.3f}, Acc={acc:.3f}')
        plt.grid(True, alpha=0.3)
        
        plt.annotate(f'ε={eps:.3f}\nAcc={acc:.3f}',
                    (eps, acc),
                    xytext=(10, 10),
                    textcoords='offset points')
        
        plt.tight_layout()
        privacy_plot = output_dir / f'privacy_tradeoff_epsilon_{epsilon}.png'
        plt.savefig(privacy_plot, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Privacy tradeoff plot saved: {privacy_plot}")


def generate_report(model_dir: Path, output_dir: Path):
    """Generate a comprehensive report."""
    output_dir.mkdir(parents=True, exist_ok=True)  # FIX: Create directory first
    
    try:
        results = load_model_results(model_dir)
        history = results['history']
        config = results['config']
        epsilon = config.get('epsilon', history.get('epsilon', 8.0))
    except Exception as e:
        print(f"⚠️  Could not load model results: {e}")
        return
    
    report = f"""
# DP Model Evaluation Report

## Model Information
- **Model Directory**: {model_dir}
- **Privacy Budget (ε)**: {epsilon}
- **Training Epochs**: {len(history.get('train_loss', []))}
- **Timestamp**: {config.get('timestamp', 'Unknown')}

## Performance Metrics
"""
    
    if 'train_loss' in history and history['train_loss']:
        report += f"- **Final Training Loss**: {history['train_loss'][-1]:.4f}\n"
    
    if 'train_acc' in history and history['train_acc']:
        report += f"- **Best Training Accuracy**: {max(history['train_acc']):.4f}\n"
    
    if 'val_acc' in history and history['val_acc']:
        report += f"- **Best Validation Accuracy**: {max(history['val_acc']):.4f}\n"
    elif 'train_accuracy' in history and history['train_accuracy']:
        report += f"- **Best Training Accuracy**: {max(history['train_accuracy']):.4f}\n"
    
    if 'privacy_cost' in history:
        pc = history['privacy_cost']
        report += f"""
## Privacy Analysis
- **Achieved ε**: {pc.get('epsilon', 'N/A'):.3f}
- **δ (delta)**: {pc.get('delta', 'N/A'):.0e}
- **Noise Multiplier**: {pc.get('sigma', 'N/A'):.3f}
- **Training Steps**: {pc.get('steps', 'N/A')}
"""
    
    report += f"""
## Files
- Checkpoint: {model_dir}/checkpoint.pkl
- History: {model_dir}/history.json
- Plots: {output_dir}/*.png
"""

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
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return
    
    print(f"📈 Evaluating model: {model_dir}")
    generate_report(model_dir, output_dir / model_dir.name)


if __name__ == "__main__":
    main()
EOF

# Also create a simple test script
cat > test_evaluation.sh << 'EOF'
#!/bin/bash
# test_evaluation.sh - Test evaluation on minimal model

echo "🧪 Testing evaluation on minimal model..."
echo "="*60

# Create evaluation directory
mkdir -p outputs/evaluation

# Run evaluation
python src/evaluation/evaluate_dp_model.py \
    --model_dir outputs/models/minimal_epsilon_8.0 \
    --output_dir outputs/evaluation

echo ""
echo "📁 Checking output files:"
ls -la outputs/evaluation/minimal_epsilon_8.0/ 2>/dev/null || echo "No evaluation output found"

echo ""
echo "🎯 If evaluation works, you should see:"
echo "   - evaluation_report.md"
echo "   - training_curves_epsilon_8.0.png"
echo "   - privacy_tradeoff_epsilon_8.0.png"
echo "="*60
EOF

chmod +x test_evaluation.sh

echo "✅ Fixed evaluation script"
echo "🚀 Test it: ./test_evaluation.sh"
