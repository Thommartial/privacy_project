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
