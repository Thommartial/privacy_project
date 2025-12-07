#!/bin/bash
# train_single.sh - Train a single epsilon value

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <epsilon> [epochs] [batch_size]"
    echo "Example: $0 2.0 5 32"
    echo "Default: epochs=5, batch_size=32"
    exit 1
fi

EPSILON=$1
EPOCHS=${2:-5}
BATCH_SIZE=${3:-32}

echo "="*60
echo "Training DP model with ε=$EPSILON"
echo "="*60

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

# Run training
python src/training/train_dp.py \
    --epsilon $EPSILON \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 5e-5 \
    --max_seq_length 128 \
    --output_dir "outputs/models" \
    --early_stopping_patience 2

# Generate plots
MODEL_NAME="distilbert_epsilon_${EPSILON}"
python src/evaluation/plot_utils.py \
    --model_dir "outputs/models/$MODEL_NAME" \
    --epsilon $EPSILON \
    --output_dir "outputs/plots"

echo ""
echo "✅ Training complete!"
echo "📁 Model saved: outputs/models/$MODEL_NAME"
echo "📈 Plots saved: outputs/plots/distilbert_epsilon_${EPSILON}/"
