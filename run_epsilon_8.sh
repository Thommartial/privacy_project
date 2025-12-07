#!/bin/bash
# run_epsilon_8.sh - Train ε=8.0 model (fastest training, best performance)

set -e

echo "="*70
echo "🚀 TRAINING ε=8.0 MODEL (Weakest Privacy, Best Performance)"
echo "="*70

# Configuration
EPSILON=8.0
MODEL_NAME="distilbert_epsilon_8.0"
EPOCHS=5
BATCH_SIZE=32
LEARNING_RATE=5e-5
MAX_SEQ_LENGTH=128
OUTPUT_DIR="outputs/models"

# Create directories
mkdir -p "$OUTPUT_DIR/$MODEL_NAME"
mkdir -p "outputs/plots/$MODEL_NAME"
mkdir -p "outputs/logs"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

# Hardware check
echo "🖥️  Hardware Status:"
python -c "
import jax
import psutil
import os

print(f'  JAX version: {jax.__version__}')
print(f'  Devices: {jax.device_count()}')
print(f'  Device type: {jax.devices()[0].device_kind}')
print(f'  Platform: {jax.default_backend()}')
print(f'  CPU cores: {psutil.cpu_count(logical=True)}')
print(f'  Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')
"

# Start training
echo ""
echo "🏋️‍♂️ Starting training for ε=$EPSILON..."
echo "   This will take ~30-45 minutes on CPU"
echo "   Press Ctrl+C to stop early"
echo "="*70

START_TIME=$(date +%s)

# Run training with progress tracking
python src/training/train_dp.py \
    --epsilon $EPSILON \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience 2

END_TIME=$(date +%s)
TRAINING_TIME=$((END_TIME - START_TIME))

# Generate plots
echo ""
echo "📊 Generating plots..."
python src/evaluation/plot_utils.py \
    --model_dir "$OUTPUT_DIR/$MODEL_NAME" \
    --epsilon $EPSILON \
    --output_dir "outputs/plots"

# Show results
echo ""
echo "="*70
echo "✅ TRAINING COMPLETE!"
echo "="*70
echo ""
echo "📋 Results for ε=$EPSILON:"
echo "   Training time: $((TRAINING_TIME / 60)) minutes $((TRAINING_TIME % 60)) seconds"
echo ""

# Show best F1 score
HISTORY_FILE="$OUTPUT_DIR/$MODEL_NAME/training_history.json"
if [[ -f "$HISTORY_FILE" ]]; then
    echo "📈 Performance metrics:"
    python -c "
import json
with open('$HISTORY_FILE') as f:
    data = json.load(f)

if 'val_f1' in data and data['val_f1']:
    best_f1 = max(data['val_f1'])
    best_epoch = data['val_f1'].index(best_f1) + 1
    print(f'   Best F1 Score: {best_f1:.4f} (epoch {best_epoch})')
    
if 'train_accuracy' in data and data['train_accuracy']:
    final_train_acc = data['train_accuracy'][-1]
    print(f'   Final Train Accuracy: {final_train_acc:.4f}')
    
if 'val_accuracy' in data and data['val_accuracy']:
    final_val_acc = data['val_accuracy'][-1]
    print(f'   Final Validation Accuracy: {final_val_acc:.4f}')

if 'privacy_cost' in data:
    pc = data['privacy_cost']
    print(f'   Privacy Cost: ε={pc[\"epsilon\"]:.3f}, δ={pc[\"delta\"]:.0e}')
"
fi

echo ""
echo "📁 Output files:"
echo "   Model checkpoint: $OUTPUT_DIR/$MODEL_NAME/model_checkpoint.pkl"
echo "   Training history: $OUTPUT_DIR/$MODEL_NAME/training_history.json"
echo "   Configuration:     $OUTPUT_DIR/$MODEL_NAME/training_config.json"
echo "   Plots:            outputs/plots/$MODEL_NAME/training_curves_epsilon_8.0.png"
echo ""

# Create next steps reminder
echo "🚀 Next steps:"
echo "   1. Check the training plots: outputs/plots/$MODEL_NAME/"
echo "   2. Run next epsilon: ./train_single.sh 5.0"
echo "   3. Or run all: ./train_all_epsilons.sh"
echo "="*70

# Show plot if possible
if command -v eog &> /dev/null && [[ -f "outputs/plots/$MODEL_NAME/training_curves_epsilon_8.0.png" ]]; then
    echo ""
    read -p "📸 Open training plot? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        eog "outputs/plots/$MODEL_NAME/training_curves_epsilon_8.0.png" &
    fi
fi
