#!/bin/bash
# run_minimal.sh - Run minimal working DP model

set -e

echo "="*60
echo "🧪 MINIMAL WORKING DP MODEL"
echo "="*60
echo "Goal: Train ANY DP model successfully"
echo "Time target: < 2 minutes"
echo "="*60

# Create output dir
mkdir -p outputs/models/minimal_epsilon_8.0

# Activate env
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

# Test import first
echo ""
echo "🔍 Testing imports..."
python test_import.py

# Check if import succeeded
if [ $? -ne 0 ]; then
    echo "❌ Import test failed"
    exit 1
fi

# Run training
echo ""
echo "🚀 Starting minimal training..."
START_TIME=$(date +%s)

python src/training/minimal_train.py \
    --epsilon 8.0 \
    --epochs 3 \
    --batch_size 8 \
    --max_seq_length 32 \
    --n_samples 100

END_TIME=$(date +%s)
TRAINING_TIME=$((END_TIME - START_TIME))

echo ""
echo "="*60
echo "📊 RESULTS"
echo "="*60
echo "Training time: ${TRAINING_TIME}s"
echo ""

# Show results
if [[ -f "outputs/models/minimal_epsilon_8.0/history.json" ]]; then
    echo "📈 Training history:"
    python -c "
import json
with open('outputs/models/minimal_epsilon_8.0/history.json') as f:
    data = json.load(f)
    
print(f'  Epochs trained: {len(data[\"train_loss\"])}')
print(f'  Final train loss: {data[\"train_loss\"][-1]:.4f}')
print(f'  Best validation accuracy: {max(data[\"val_acc\"]):.4f}')
if 'privacy_cost' in data:
    print(f'  Privacy cost: ε={data[\"privacy_cost\"][\"epsilon\"]:.3f}')
"
fi

echo ""
echo "🎯 NEXT STEPS:"
echo "   1. This proves DP training WORKS"
echo "   2. Now we can make the model more complex"
echo "   3. Run: python src/training/minimal_train.py --epochs 5 --n_samples 500"
echo "="*60
