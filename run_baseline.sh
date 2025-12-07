#!/bin/bash
# run_baseline.sh - Run baseline evaluation

echo "="*50
echo "🎯 PHASE 3: BASELINE EVALUATION"
echo "="*50

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate pii-jax

# Run baseline evaluation
python src/baseline/template_baseline.py

echo ""
echo "✅ Baseline evaluation complete!"
echo "📁 Results in outputs/results/"
ls -la outputs/results/
