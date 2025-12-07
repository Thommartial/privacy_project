#!/bin/bash
# run_dp_training.sh - Train DP model with ε=8 (Fast & Efficient)

set -e  # Exit on error

echo "="*70
echo "🚀 PHASE 4: DP MODEL TRAINING (ε = 8)"
echo "="*70

# Configuration
EPSILON=8.0
MODEL_NAME="distilbert_epsilon_8"
EPOCHS=5
BATCH_SIZE=32
LEARNING_RATE=5e-5
MAX_SEQ_LENGTH=128
OUTPUT_DIR="outputs/models/$MODEL_NAME"

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p outputs/plots/$MODEL_NAME
mkdir -p outputs/logs/$MODEL_NAME

echo "📋 Configuration:"
echo "   ε (epsilon): $EPSILON"
echo "   Model: $MODEL_NAME"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Output: $OUTPUT_DIR"
echo "="*70

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

# Check GPU availability
echo "🖥️  Hardware check:"
python -c "
import jax
print(f'  JAX devices: {jax.device_count()}')
print(f'  Device type: {jax.devices()[0].device_kind}')
print(f'  Platform: {jax.default_backend()}')
"

# Run training
echo ""
echo "🏋️‍♂️ Starting training..."
python src/training/train_dp.py \
    --epsilon $EPSILON \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir $OUTPUT_DIR \
    --early_stopping_patience 2

# Generate plots
echo ""
echo "📊 Generating evaluation plots..."
python src/evaluation/generate_plots.py \
    --model_dir $OUTPUT_DIR \
    --epsilon $EPSILON \
    --output_dir outputs/plots/$MODEL_NAME

echo ""
echo "="*70
echo "✅ TRAINING COMPLETE!"
echo "="*70
echo ""
echo "📁 Output files:"
echo "   Model checkpoints: $OUTPUT_DIR/"
echo "   Training logs: outputs/logs/$MODEL_NAME/"
echo "   Evaluation plots: outputs/plots/$MODEL_NAME/"
echo ""
echo "📊 Next: Run evaluation with:"
echo "   python src/evaluation/evaluate_model.py --model_dir $OUTPUT_DIR"
echo "="*70
