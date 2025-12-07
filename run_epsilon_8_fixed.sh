#!/bin/bash
# run_epsilon_8_fixed.sh - Train ε=8.0 with working model

set -e

echo "="*70
echo "🚀 TRAINING ε=8.0 MODEL (Fixed Version)"
echo "="*70

# Configuration - SMALLER for faster debugging
EPSILON=8.0
MODEL_NAME="distilbert_epsilon_8.0"
EPOCHS=3
BATCH_SIZE=8
LEARNING_RATE=5e-5
MAX_SEQ_LENGTH=32  # Very small for debugging
OUTPUT_DIR="outputs/models"

# Create directories
mkdir -p "$OUTPUT_DIR/$MODEL_NAME"
mkdir -p "outputs/plots/$MODEL_NAME"

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

echo "🖥️  Hardware Status:"
python -c "
import jax
print(f'  JAX version: {jax.__version__}')
print(f'  Devices: {jax.device_count()}')
"

echo ""
echo "🏋️‍♂️ Starting training for ε=$EPSILON..."
echo "   Batch size: $BATCH_SIZE, Seq length: $MAX_SEQ_LENGTH"
echo "   Epochs: $EPOCHS"
echo "="*70

START_TIME=$(date +%s)

# Run training with DEBUG mode
python -c "
import sys
import os
sys.path.insert(0, os.getcwd())

print('Testing model import...')
try:
    from src.models.distilbert_dp import DistilBertDP
    print('✅ Model imports OK')
    
    import jax
    import jax.numpy as jnp
    
    # Test model initialization
    model = DistilBertDP(num_hidden_layers=1, hidden_size=128)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((2, 32), dtype=jnp.int32)
    dummy_mask = jnp.ones((2, 32), dtype=jnp.int32)
    
    params = model.init(rng, dummy_input, dummy_mask, training=False)
    print(f'✅ Model initialized successfully')
    print(f'   Parameter count: {sum(x.size for x in jax.tree_util.tree_leaves(params)):,}')
    
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
"

# If model test passes, run training
echo ""
echo "🚀 Running actual training..."
python src/training/train_dp.py \
    --epsilon $EPSILON \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience 1

END_TIME=$(date +%s)
TRAINING_TIME=$((END_TIME - START_TIME))

echo ""
echo "="*70
echo "✅ TRAINING COMPLETE!"
echo "="*70
echo ""
echo "📋 Results:"
echo "   Training time: $((TRAINING_TIME / 60))m $((TRAINING_TIME % 60))s"
echo "   Model saved: $OUTPUT_DIR/$MODEL_NAME/"
echo ""

# Check results
if [[ -f "$OUTPUT_DIR/$MODEL_NAME/training_history.json" ]]; then
    echo "📊 Training metrics:"
    python -c "
import json
with open('$OUTPUT_DIR/$MODEL_NAME/training_history.json') as f:
    data = json.load(f)
    
if 'val_f1' in data and data['val_f1']:
    best_f1 = max(data['val_f1'])
    print(f'   Best F1 Score: {best_f1:.4f}')
    
if 'train_accuracy' in data and data['train_accuracy']:
    final_acc = data['train_accuracy'][-1]
    print(f'   Final Train Accuracy: {final_acc:.4f}')
"
fi

echo ""
echo "🎯 Next: If this works, increase parameters for better results:"
echo "   ./train_single.sh 8.0 5 16 64  # ε=8.0, 5 epochs, batch 16, seq 64"
echo "="*70
